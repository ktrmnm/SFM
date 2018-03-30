#ifndef SET_UTILS_H
#define SET_UTILS_H

#include <vector>
#include <utility>
#include <string>
#include <stdexcept>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
//#include <initializer_list>

namespace submodular {

class Set;
class Partition;
bool operator == (const Set& lhs, const Set& rhs);
bool operator != (const Set& lhs, const Set& rhs);
std::ostream& operator << (std::ostream& stream, const Set& X);

using element_type = std::size_t;

class Set {
public:
  // Default constructor
  Set(): n_(0), bits_() {}

  Set(const Set&) = default;
  Set(Set&&) = default;

  // Constructor (a): Construct a Set object just from its size.
  // All elements of bits_ are initialized by 0.
  explicit Set(std::size_t n): n_(n), bits_(n, 0) {}

  // Constructor (b): Construct a Set object from its size and a set of indices
  // in which bits_[i] == 1. If vec contains an index that is larger than n - 1,
  // it throws std::out_of_range.
  //explicit Set(std::size_t n, std::vector<std::size_t> vec);
  //explicit Set(std::size_t n, std::initializer_list<std::size_t> vec);

  // Constructor (c): Construct a Set object from a string object consists of
  // 0-1 characters. If any characters examined in str are not 0 or 1, it throws
  // std::invalid_argument.
  template<class CharT>
  explicit Set(const std::basic_string<CharT>& str);

  // Constructor (d): Construct a Set from a bitmask.
  // The first M bits are initialized by the corresponding bit values of val,
  // where M is the number of radix digits of unsigned long type (typically 32 or 64).
  // If n > M, the remainders are filled by 0.
  explicit Set(std::size_t n, unsigned long val);

  Set& operator=(const Set&) = default;
  Set& operator=(Set&&) = default;

  // static factory methods
  static Set MakeDense(std::size_t n);
  static Set MakeEmpty(std::size_t n);
  static Set FromIndices(std::size_t n, const std::vector<element_type>& indices);

  bool HasElement(element_type i) const;
  bool operator[] (element_type i) const { return HasElement(i); }

  std::vector<element_type> GetMembers() const;
  std::vector<std::size_t> GetInverseMap() const;
  std::size_t Cardinality() const;
  void AddElement(element_type i);
  void RemoveElement(element_type i);

  // Returns a new copy
  Set Copy() const;

  // Returns a new copy of the complement
  Set Complement() const;
  Set C() const { return Complement(); }
  //Set operator~() const;

  Set Union(const Set& X) const;
  Set Intersection(const Set& X) const;

  std::size_t n_;
  std::vector<char> bits_;
  friend std::ostream& operator << (std::ostream&, const Set&);
};

class Partition {
public:
  Partition() = default;
  Partition(const Partition&) = default;
  Partition(Partition&&) = default;
  Partition& operator=(const Partition&) = default;
  Partition& operator=(Partition&&) = default;

  // static method to generate the finest partition {{0}, {1}, ..., {n-1}}
  static Partition MakeFine(std::size_t n);
  // static method to generate the coarsest partition {{0, 1, ..., n-1}}
  //static Partition MakeCoarse(std::size_t n);

  bool HasCell(std::size_t cell) const;
  std::vector<std::size_t> GetCellIndices() const;
  std::size_t Cardinality() const;

  // Return a Set object placed at the given cell.
  // If cell is not found, it returns an empty
  Set GetCellAsSet(std::size_t cell) const;

  void RemoveCell(std::size_t cell);

  // Merge cells
  // If merge operation successed, then returns the new cell index.
  // Otherwise, just returns the minimum existing cell index or n_
  std::size_t MergeCells(std::size_t cell_1, std::size_t cell_2);
  std::size_t MergeCells(const std::vector<std::size_t>& cells);

  Set Expand() const;
  Set Expand(const Set& X) const;

  // (possible largest index of elements) + 1. Do not confuse with Cardinality().
  std::size_t n_;

  // pairs of {representable elements, cell members}
  // Cell members are maintained to be sorted as long as change operations are performed
  // only through the member functions.
  std::unordered_map<std::size_t, std::vector<std::size_t>> cells_;
};


/*
Set::Set(std::size_t n, std::vector<std::size_t> vec)
  : n_(n), bits_(n, 0)
{
  for (const auto& i: vec) {
    if (i >= n) {
      throw std::range_error("Set::constructor_b");
    }
    else {
      bits_[i] = 1;
    }
  }
}
*/

Set Set::FromIndices(std::size_t n, const std::vector<element_type>& indices) {
  Set X(n);
  for (const auto& i: indices) {
    if (i >= n) {
      throw std::range_error("Set::FromIndices");
    }
    else {
      X.bits_[i] = 1;
    }
  }
  return X;
}

template<class CharT>
Set::Set(const std::basic_string<CharT>& str)
  : n_(str.size()), bits_(str.size(), 0)
{
  for (std::size_t i = 0; i < n_; ++i) {
    if (str[i] == CharT('0')) {
      bits_[i] = 0;
    }
    else if (str[i] == CharT('1')) {
      bits_[i] = 1;
    }
    else {
      throw std::invalid_argument("Set::constructor_c");
    }
  }
}

Set::Set(std::size_t n, unsigned long val)
  : n_(n), bits_(n, 0)
{
  for (std::size_t i = 0; i < n; ++i) {
    bits_[i] = (val >> i & 1) == 1 ? 1 : 0;
  }
}

bool Set::HasElement(element_type i) const {
  return static_cast<bool>(bits_[i]);
}

Set Set::Copy() const {
  Set X; X.n_ = n_; X.bits_ = bits_;
  return X;
}

Set Set::Complement() const{
  Set X = Copy();
  for (auto& b: X.bits_) {
    b = !b;
  }
  return X;
}

Set Set::Union(const Set& X) const {
  Set X_new(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    if (X.HasElement(i) || this->HasElement(i)) {
      X_new.bits_[i] = 1;
    }
  }
  return X_new;
}

Set Set::Intersection(const Set& X) const {
  Set X_new(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    if (X.HasElement(i) && this->HasElement(i)) {
      X_new.bits_[i] = 1;
    }
  }
  return X_new;
}

std::vector<element_type> Set::GetMembers() const {
  std::vector<element_type> members;
  for (std::size_t i = 0; i < n_; ++i) {
    if (HasElement(i)) {
      members.push_back(i);
    }
  }
  return members;
}

// NOTE: this is used as follows:
// auto members = X.GetMembers(); auto inverse = X.GetInverseMap();
// for (element_type i: members) {
//   assert(i == members[inverse[i]]);
// }
std::vector<std::size_t> Set::GetInverseMap() const {
  std::vector<std::size_t> inverse(n_, n_);
  std::size_t pos = 0;
  for (std::size_t i = 0; i < n_; ++i) {
    if (HasElement(i)) {
      inverse[i] = pos;
      pos++;
    }
  }
  return inverse;
}

std::size_t Set::Cardinality() const {
  std::size_t c = 0;
  for (const auto& b: bits_) {
    if (b) { c++; }
  }
  return c;
}

void Set::AddElement(element_type i) {
  bits_.at(i) = 1;
}

void Set::RemoveElement(element_type i) {
  bits_.at(i) = 0;
}

Set Set::MakeDense(std::size_t n) {
  Set X;
  X.n_ = n;
  X.bits_ = std::vector<char>(n, 1);
  return X;
}

Set Set::MakeEmpty(std::size_t n) {
  Set X(n);
  return X;
}

std::ostream& operator << (std::ostream& stream, const Set& X) {
  auto members = X.GetMembers();
  stream << "{";
  for (std::size_t pos = 0; pos < members.size(); ++pos) {
    stream << members[pos];
    if (pos < members.size() - 1) {
      stream << ", ";
    }
  }
  stream << "}";
  return stream;
}

bool operator == (const Set& lhs, const Set& rhs) {
  if (lhs.n_ != rhs.n_) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.n_; ++i) {
    if ((bool)lhs.bits_[i] != (bool)rhs.bits_[i]) {
      return false;
    }
  }
  return true;
}

bool operator != (const Set& lhs, const Set& rhs) {
  return !(lhs == rhs);
}

Partition Partition::MakeFine(std::size_t n) {
  Partition p;
  p.n_ = n;
  for (std::size_t cell = 0; cell < n; ++cell) {
    p.cells_[cell] = std::vector<std::size_t>(1, cell);
  }
  return p;
}

bool Partition::HasCell(std::size_t cell) const {
  return (cells_.count(cell) == 1);
}

std::vector<std::size_t> Partition::GetCellIndices() const {
  std::vector<std::size_t> indices;
  for (std::size_t cell = 0; cell < n_; ++cell) {
    if (cells_.count(cell) == 1) {
      indices.push_back(cell);
    }
  }
  return indices;
}

std::size_t Partition::Cardinality() const {
  std::size_t card = 0;
  for (const auto& kv: cells_) {
    card += kv.second.size();
  }
  return card;
}

Set Partition::GetCellAsSet(std::size_t cell) const {
  if (HasCell(cell)) {
    auto indices = cells_.at(cell);
    return Set::FromIndices(n_, indices);
  }
  else {
    return Set::MakeEmpty(n_);
  }
}

void Partition::RemoveCell(std::size_t cell) {
  if (HasCell(cell)) {
    cells_.erase(cells_.find(cell));
  }
}

std::size_t Partition::MergeCells(std::size_t cell_1, std::size_t cell_2) {
  auto c1 = HasCell(cell_1) ? cell_1 : n_;
  auto c2 = HasCell(cell_2) ? cell_2 : n_;
  auto cell_min = std::min(c1, c2);

  if (c1 < n_ && c2 < n_ && cell_1 != cell_2) {
    auto cell_max = std::max(cell_1, cell_2);
    auto indices_max = cells_[cell_max];
    cells_[cell_min].reserve(cells_[cell_min].size() + indices_max.size());
    for (const auto& i: indices_max) {
      cells_[cell_min].push_back(i);
    }
    std::sort(cells_[cell_min].begin(), cells_[cell_min].end());
    cells_.erase(cells_.find(cell_max));
  }

  return cell_min;
}

std::size_t Partition::MergeCells(const std::vector<std::size_t>& cells) {
  std::unordered_set<std::size_t> cells_to_merge;
  std::size_t cell_new = n_;
  for (const auto& cell: cells) {
    if (cell < cell_new) {
      cell_new = cell;
    }
    if (HasCell(cell)) {
      cells_to_merge.insert(cell);
    }
  }
  for (const auto& cell: cells_to_merge) {
    if (cell != cell_new) {
      for (const auto& i: cells_[cell]) {
        cells_[cell_new].push_back(i);
      }
      cells_.erase(cells_.find(cell));
    }
  }
  std::sort(cells_[cell_new].begin(), cells_[cell_new].end());
  return cell_new;
}

Set Partition::Expand() const {
  Set expanded = Set::MakeEmpty(n_);
  for (const auto& kv: cells_){
    auto indices = kv.second;
    for (const auto& i: indices) {
      expanded.bits_[i] = 1;
    }
  }
  return expanded;
}

Set Partition::Expand(const Set& X) const {
  Set expanded = Set::MakeEmpty(n_);
  for (const auto& kv: cells_) {
    auto cell = kv.first;
    if (X[cell]) {
      auto indices = kv.second;
      for (const auto& i: indices) {
        expanded.bits_[i] = 1;
      }
    }
  }
  return expanded;
}

}//namespace submodular

#endif
