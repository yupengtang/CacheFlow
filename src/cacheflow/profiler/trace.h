#pragma once

#include "cacheflow/common.h"
#include <string>
#include <vector>
#include <fstream>
#include <mutex>

namespace cacheflow {

// ── Chrome Trace Event format for visualization ─────────────────────────────

struct TraceEvent {
    std::string name;
    std::string category;
    TimePoint   timestamp;
    Duration    duration{0};
    int32_t     thread_id  = 0;
    int32_t     process_id = 0;

    enum class Phase : char {
        BEGIN     = 'B',
        END       = 'E',
        COMPLETE  = 'X',
        INSTANT   = 'i',
        COUNTER   = 'C',
    };
    Phase phase = Phase::COMPLETE;

    struct Arg {
        std::string key;
        std::string value;
    };
    std::vector<Arg> args;
};

// ── Trace writer: outputs Chrome Trace JSON ─────────────────────────────────

class TraceWriter {
public:
    explicit TraceWriter(const std::string& output_path);
    ~TraceWriter();

    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

    void add_event(TraceEvent event);
    void add_counter(const std::string& name,
                     const std::string& series,
                     double value);
    void flush();

    size_t num_events() const;

private:
    void write_event(const TraceEvent& event);

    std::string output_path_;
    std::ofstream file_;
    std::vector<TraceEvent> events_;
    mutable std::mutex mu_;
    TimePoint start_time_;
    bool first_event_ = true;
};

// ── Scoped trace span ───────────────────────────────────────────────────────

class TraceSpan {
public:
    TraceSpan(TraceWriter* writer, std::string name,
              std::string category = "");
    ~TraceSpan();

    TraceSpan(const TraceSpan&) = delete;
    TraceSpan& operator=(const TraceSpan&) = delete;

    void add_arg(const std::string& key, const std::string& value);

private:
    TraceWriter* writer_;
    TraceEvent   event_;
};

}  // namespace cacheflow
