#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <ctime>

using namespace std;

template <typename T>
struct TimestampValue {
    T value;
    int stamp;

    TimestampValue() : value(T()), stamp(0) {}
    TimestampValue(T val, int st) : value(val), stamp(st) {}
};

//function to get current time
string getCurrentTimestamp() {
    using namespace chrono;
    auto now = system_clock::now();
    time_t now_c = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    tm localTime = *localtime(&now_c);

    ostringstream oss;
    oss << put_time(&localTime, "%H:%M:%S") << ":" << setfill('0') << setw(3) << ms.count();
    return oss.str();
}

// Wait-Free Snapshot class
class MRMW_Snapshot {
public:
    MRMW_Snapshot(int size) : snap_size(size), array(size) {
        for (int i = 0; i < size; ++i) {
            array[i].store(TimestampValue<int>(0, 0));
        }
    }

    void update(int loc, int value) {
        TimestampValue<int> oldValue = array[loc].load();
        TimestampValue<int> newValue(value, oldValue.stamp + 1);
        array[loc].store(newValue); 
    }

    vector<int> snapshot() {
        vector<TimestampValue<int>> localSnapshot(snap_size);
        int maxStamp = -1;

        // Recording the current state 
        for (int i = 0; i < snap_size; ++i) {
            localSnapshot[i] = array[i].load();
            maxStamp = max(maxStamp, localSnapshot[i].stamp);
        }

        
        for (int i = 0; i < snap_size; ++i) {
            TimestampValue<int> currentValue = array[i].load();
            if (currentValue.stamp > maxStamp) {
                
                return snapshot(); 
            }
        }

        return taketheSanpshot(localSnapshot); // Returns snapshot values
    }

private:
    int snap_size;
    vector<atomic<TimestampValue<int>>> array;

    vector<int> taketheSanpshot(const vector<TimestampValue<int>>& snap) {
        vector<int> result(snap_size);
        for (int i = 0; i < snap_size; ++i) {
            result[i] = snap[i].value;
        }
        return result;
    }
};

// Writer thread function
void writer(MRMW_Snapshot& snap_obj, atomic<bool>& term, int thread_id, int M, double mu_w, 
            vector<long long>& update_times, long long& max_update_time, ofstream& log_file) {
    random_device rd;
    mt19937 gen(rd());
    exponential_distribution<> dist(mu_w);

    while (!term) {
        int value = rand() % 100;
        int loc = rand() % M;

        auto start = chrono::high_resolution_clock::now();
        snap_obj.update(loc, value);
        auto end = chrono::high_resolution_clock::now();
        auto time_span = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        update_times.push_back(time_span);

        // Updating maximum update time
        if (time_span > max_update_time) {
            max_update_time = time_span;
        }

        
        log_file << getCurrentTimestamp() << " - Thread " << thread_id << " wrote " << value 
                 << " at location " << loc<<endl;

        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(dist(gen))));
    }
}

// Snapshot thread function
void snapshot_thread(MRMW_Snapshot& snap_obj, int k, double mu_s, 
                     vector<long long>& snapshot_times, long long& max_snapshot_time, ofstream& log_file) {
    random_device rd;
    mt19937 gen(rd());
    exponential_distribution<> dist(mu_s);

    for (int i = 0; i < k; ++i) {
        auto start = chrono::high_resolution_clock::now();
        vector<int> snap = snap_obj.snapshot();  
        auto end = chrono::high_resolution_clock::now();
        auto time_span = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        snapshot_times.push_back(time_span); 

        // Updating maximum snapshot time
        if (time_span > max_snapshot_time) {
            max_snapshot_time = time_span;
        }

        
        log_file << getCurrentTimestamp() << " - Snapshot " << i + 1 << " collected in ";
        for (int val : snap) {
            log_file << val<< " ";
        }
        log_file << endl;

        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(dist(gen))));
    }
}

int main() {
    ifstream input("inp-params.txt");
    int nw, ns, M, k;
    double mu_w, mu_s;

    if (!input.is_open()) {
        cerr << "Error opening inp-params.txt" << endl;
        return 1;
    }

    input >> nw >> ns >> M >> mu_w >> mu_s >> k;

    MRMW_Snapshot snap_obj(M);
    atomic<bool> term(false);

    vector<thread> writers, snapshotters;
    vector<vector<long long>> update_times(nw);   
    vector<vector<long long>> snapshot_times(ns);

    // Variables for worst-case tracking
    long long max_update_time = 0;  
    long long max_snapshot_time = 0; 

    ofstream log_file("output.txt");

    // Creating writer threads
    for (int i = 0; i < nw; ++i) {
        writers.emplace_back(writer, ref(snap_obj), ref(term), i, M, mu_w, ref(update_times[i]), ref(max_update_time), ref(log_file));
    }

    // Creating snapshot threads
    for (int i = 0; i < ns; ++i) {
        snapshotters.emplace_back(snapshot_thread, ref(snap_obj), k, mu_s, ref(snapshot_times[i]), ref(max_snapshot_time), ref(log_file));
    }

    // Waiting for all snapshot threads to finish
    for (auto& snapper : snapshotters) {
        snapper.join();
    }
    term = true;

    for (auto& writer : writers) {
        writer.join();
    }

    // Calculating the average update time
    long long total_update_time = 0;
    int total_updates = 0;

    for (const auto& times : update_times) {
        for (long long time : times) {
            total_update_time += time;
            total_updates++;
        }
    }

    double average_update_time = (total_updates > 0) ? static_cast<double>(total_update_time) / total_updates : 0;

    // Calculating the average snapshot time
    long long t_snapshot_time = 0;
    int t_snapshots = 0;

    for (const auto& times : snapshot_times) {
        for (long long time : times) {
            t_snapshot_time += time;
            t_snapshots++;
        }
    }

    double average_snapshot_time = (t_snapshots > 0) ? static_cast<double>(t_snapshot_time) / t_snapshots : 0;

    // To get output to the console
    cout << "Average update thread time: " << average_update_time <<endl;
    cout << "Average snapshot thread time: " << average_snapshot_time <<endl;
    cout << "Worst-case update thread time: " << max_update_time <<endl;
    cout << "Worst-case snapshot thread time: " << max_snapshot_time <<endl;

    // To get output to the log file
    log_file << "Average update thread time: " << average_update_time <<endl;
    log_file << "Average snapshot thread time: " << average_snapshot_time <<endl;
    log_file << "Worst-case update thread time: " << max_update_time <<endl;
    log_file << "Worst-case snapshot thread time: " << max_snapshot_time << endl;
    

    log_file.close();

    return 0;
}

