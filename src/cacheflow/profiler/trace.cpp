#include "cacheflow/profiler/trace.h"
#include <sstream>
#include <iomanip>
#include <thread>

namespace cacheflow {

TraceWriter::TraceWriter(const std::string& output_path)
    : output_path_(output_path)
    , start_time_(Clock::now())
{
    file_.open(output_path_);
    if (file_.is_open()) {
        file_ << "[\n";
    }
}

TraceWriter::~TraceWriter() {
    flush();
    if (file_.is_open()) {
        file_ << "\n]\n";
        file_.close();
    }
}

void TraceWriter::add_event(TraceEvent event) {
    std::lock_guard<std::mutex> lock(mu_);
    if (event.thread_id == 0) {
        std::hash<std::thread::id> hasher;
        event.thread_id = static_cast<int32_t>(
            hasher(std::this_thread::get_id()) & 0xFFFF);
    }
    events_.push_back(std::move(event));

    if (events_.size() >= 1000) {
        for (auto& ev : events_)
            write_event(ev);
        events_.clear();
    }
}

void TraceWriter::add_counter(const std::string& name,
                              const std::string& series,
                              double value) {
    TraceEvent ev;
    ev.name      = name;
    ev.category  = "counter";
    ev.phase     = TraceEvent::Phase::COUNTER;
    ev.timestamp = Clock::now();
    ev.args.push_back({series, std::to_string(value)});
    add_event(std::move(ev));
}

void TraceWriter::flush() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& ev : events_)
        write_event(ev);
    events_.clear();
    if (file_.is_open())
        file_.flush();
}

size_t TraceWriter::num_events() const {
    std::lock_guard<std::mutex> lock(mu_);
    return events_.size();
}

void TraceWriter::write_event(const TraceEvent& event) {
    if (!file_.is_open()) return;

    double ts_us = std::chrono::duration<double, std::micro>(
        event.timestamp - start_time_).count();
    double dur_us = std::chrono::duration<double, std::micro>(
        event.duration).count();

    if (!first_event_) file_ << ",\n";
    first_event_ = false;

    file_ << "  {\"name\":\"" << event.name
          << "\",\"cat\":\"" << event.category
          << "\",\"ph\":\"" << static_cast<char>(event.phase)
          << "\",\"ts\":" << std::fixed << std::setprecision(1) << ts_us
          << ",\"dur\":" << std::fixed << std::setprecision(1) << dur_us
          << ",\"pid\":" << event.process_id
          << ",\"tid\":" << event.thread_id;

    if (!event.args.empty()) {
        file_ << ",\"args\":{";
        for (size_t i = 0; i < event.args.size(); ++i) {
            if (i > 0) file_ << ",";
            file_ << "\"" << event.args[i].key << "\":\""
                  << event.args[i].value << "\"";
        }
        file_ << "}";
    }

    file_ << "}";
}

// ── TraceSpan ───────────────────────────────────────────────────────────────

TraceSpan::TraceSpan(TraceWriter* writer, std::string name,
                     std::string category)
    : writer_(writer)
{
    event_.name      = std::move(name);
    event_.category  = std::move(category);
    event_.phase     = TraceEvent::Phase::COMPLETE;
    event_.timestamp = Clock::now();
}

TraceSpan::~TraceSpan() {
    if (writer_) {
        event_.duration = std::chrono::duration_cast<Duration>(
            Clock::now() - event_.timestamp);
        writer_->add_event(std::move(event_));
    }
}

void TraceSpan::add_arg(const std::string& key, const std::string& value) {
    event_.args.push_back({key, value});
}

}  // namespace cacheflow
