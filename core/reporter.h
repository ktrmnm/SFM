#ifndef REPORTER_H
#define REPORTER_H

#include <string>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <utility>
#include "core/set_utils.h"

namespace submodular {

struct SimpleTimer {
  static auto Now() { return std::chrono::system_clock::now(); }
};

enum class ReportKind {
  ORACLE,
  BASE,
  TOTAL,
  ITERATION
};

class SFMReporter;

std::ostream& operator << (std::ostream& stream, const SFMReporter& reporter);

class SFMReporter {
public:
  SFMReporter() = default;

  using duration_type = typename std::chrono::milliseconds;
  const std::string duration_suffix_ = "msec";

  std::unordered_map<ReportKind, unsigned int> counts_;
  std::unordered_map<ReportKind, duration_type> times_;

  std::string algorithm_name_;
  std::string oracle_name_;
  double minimum_value_;
  Set minimizer_;

  void Clear();

  void EntryCounter(ReportKind kind);
  void EntryTimer(ReportKind kind);
  void IncreaseCount(ReportKind kind, unsigned int num = 1);
  void TimerStart(ReportKind kind);
  void TimerStop(ReportKind kind);

  void SetNames(std::string alg_name, std::string oracle_name);
  void SetResults(double minimum_value, const Set& minimizer);
  void SetResults(double minimum_value, Set&& minimizer);

  friend std::ostream& operator << (std::ostream&, const SFMReporter&);
private:
  std::unordered_map<ReportKind, bool> start_flags_;
  std::unordered_map<ReportKind, decltype(SimpleTimer::Now())> start_times_;
};

void SFMReporter::Clear() {
  for (auto& kv: times_) {
    kv.second = duration_type::zero();
  }
  for (auto& kv: counts_) {
    kv.second = 0;
  }
  for (auto& kv: start_flags_) {
    kv.second = false;
  }
}

void SFMReporter::EntryCounter(ReportKind kind) {
  counts_.insert({ kind, 0 });
}

void SFMReporter::EntryTimer(ReportKind kind) {
  times_.insert({ kind, duration_type::zero() });
  start_flags_.insert({ kind, false });
  start_times_.insert({ kind, SimpleTimer::Now() });
}

void SFMReporter::IncreaseCount(ReportKind kind, unsigned int num) {
  counts_[kind] += num;
}

void SFMReporter::TimerStart(ReportKind kind) {
  start_flags_[kind] = true;
  start_times_[kind] = SimpleTimer::Now();
}

void SFMReporter::TimerStop(ReportKind kind) {
  if (start_flags_[kind]) {
    auto end = SimpleTimer::Now();
    auto start = start_times_[kind];
    times_[kind] += std::chrono::duration_cast<duration_type>(end - start);
    start_flags_[kind] = false;
  }
}

void SFMReporter::SetNames(std::string alg_name, std::string oracle_name) {
  algorithm_name_ = alg_name;
  oracle_name_ = oracle_name;
}

void SFMReporter::SetResults(double minimum_value, const Set& minimizer) {
  minimum_value_ = minimum_value;
  minimizer_ = minimizer;
}

void SFMReporter::SetResults(double minimum_value, Set&& minimizer) {
  minimum_value_ = minimum_value;
  minimizer_ = std::move(minimizer);
}

std::ostream& operator << (std::ostream& stream, const SFMReporter& reporter) {
  stream << "Algorithm: " << reporter.algorithm_name_ << '\n';
  stream << "Oracle: " << reporter.oracle_name_ << '\n';
  stream << "Minimum value: " << reporter.minimum_value_ << '\n';
  stream << "Minimizer:\n";
  stream << reporter.minimizer_ << '\n';
  if (reporter.times_.count(ReportKind::TOTAL) == 1) {
    stream << "Total time: " << reporter.times_.at(ReportKind::TOTAL).count() << reporter.duration_suffix_ << '\n';
  }
  if (reporter.counts_.count(ReportKind::ORACLE) == 1) {
    stream << "Oracle calls: " << reporter.counts_.at(ReportKind::ORACLE) << '\n';
  }
  if (reporter.times_.count(ReportKind::ORACLE) == 1) {
    stream << "Oracle time: " << reporter.times_.at(ReportKind::ORACLE).count() << reporter.duration_suffix_ << '\n';
  }
  if (reporter.counts_.count(ReportKind::BASE) == 1) {
    stream << "Base calls: " << reporter.counts_.at(ReportKind::BASE) << '\n';
  }
  if (reporter.times_.count(ReportKind::BASE) == 1) {
    stream << "Base time: " << reporter.times_.at(ReportKind::BASE).count() << reporter.duration_suffix_ << '\n';
  }
  if (reporter.counts_.count(ReportKind::ITERATION) == 1) {
    stream << "Iterations: " << reporter.counts_.at(ReportKind::ITERATION) << '\n';
  }

  return stream;
}

}

#endif
