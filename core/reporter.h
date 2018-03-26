#ifndef REPORTER_H
#define REPORTER_H

#include <string>
#include <unordered_map>
#include <chrono>

namespace submodular {

struct SimpleTimer {
  static auto Now() { return std::chrono::system_clock::now(); }
};

enum class ReportKind {
  ORACLE,
  BASE,
  TOTAL
};

class SFMReporter {
public:
  SFMReporter() = default;

  using duration_type = typename std::chrono::milliseconds;

  std::unordered_map<ReportKind, unsigned int> counts;
  std::unordered_map<ReportKind, duration_type> times;

  void Clear();

  void EntryCounter(ReportKind kind);
  void EntryTimer(ReportKind kind);
  void IncreaseCount(ReportKind kind, unsigned int num = 1);
  void TimerStart(ReportKind kind);
  void TimerStop(ReportKind kind);

private:
  std::unordered_map<ReportKind, bool> start_flags_;
  std::unordered_map<ReportKind, decltype(SimpleTimer::Now())> start_times_;
};

void SFMReporter::Clear() {
  for (auto& kv: times) {
    kv.second = duration_type::zero();
  }
  for (auto& kv: counts) {
    kv.second = 0;
  }
  for (auto& kv: start_flags_) {
    kv.second = false;
  }
}

void SFMReporter::EntryCounter(ReportKind kind) {
  counts.insert({ kind, 0 });
}

void SFMReporter::EntryTimer(ReportKind kind) {
  times.insert({ kind, duration_type::zero() });
  start_flags_.insert({ kind, false });
  start_times_.insert({ kind, SimpleTimer::Now() });
}

void SFMReporter::IncreaseCount(ReportKind kind, unsigned int num) {
  counts[kind] += num;
}

void SFMReporter::TimerStart(ReportKind kind) {
  start_flags_[kind] = true;
  start_times_[kind] = SimpleTimer::Now();
}

void SFMReporter::TimerStop(ReportKind kind) {
  if (start_flags_[kind]) {
    auto end = SimpleTimer::Now();
    auto start = start_times_[kind];
    times[kind] += std::chrono::duration_cast<duration_type>(end - start);
    start_flags_[kind] = false;
  }
}

}

#endif
