#ifndef SET_UTILS_H
#define SET_UTILS_H

#include <vector>
#include <utility>
#include <string>
#include <stdexcept>
//#include <initializer_list>

namespace submodular {

class Set;
bool operator == (const Set& lhs, const Set& rhs);
bool operator != (const Set& lhs, const Set& rhs);

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
  explicit Set(std::size_t n, std::vector<std::size_t> vec);
  //explicit Set(std::size_t n, std::initializer_list<std::size_t> vec);

  // Constructor (c): Construct a Set object from a string object consists of
  // 0-1 characters. If any characters examined in str are not 0 or 1, it throws
  // std::invalid_argument.
  template<class CharT>
  explicit Set(const std::basic_string<CharT>& str);

  Set& operator=(const Set&) = default;
  Set& operator=(Set&&) = default;

  // static factory methods
  static Set MakeDense(std::size_t n);
  static Set MakeEmpty(std::size_t n);

  bool HasElement(std::size_t pos) const;
  bool operator[] (std::size_t pos) const { return HasElement(pos); }

  std::vector<std::size_t> GetMembers() const;
  std::size_t Cardinality() const;
  void AddElement(std::size_t pos);

  // Negate the all bits
  void Complement();
  void C() { Complement(); }

  // Returns a new copy
  Set Copy() const;
  // Returns a new copy of the complement
  Set operator~() const;

  Set Union(const Set& X) const;
  Set Intersection(const Set& X) const;

  std::size_t n_;
  std::vector<char> bits_;
};

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

bool Set::HasElement(std::size_t pos) const {
  return static_cast<bool>(bits_[pos]);
}

void Set::Complement() {
  for (auto& b: bits_) {
    b = !b;
  }
}

Set Set::Copy() const {
  Set X; X.n_ = n_; X.bits_ = bits_;
  return X;
}

Set Set::operator~() const {
  Set X = Copy();
  X.Complement();
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

std::vector<std::size_t> Set::GetMembers() const {
  std::vector<std::size_t> members;
  for (std::size_t i = 0; i < n_; ++i) {
    if (HasElement(i)) {
      members.push_back(i);
    }
  }
  return members;
}

std::size_t Set::Cardinality() const {
  std::size_t c = 0;
  for (const auto& b: bits_) {
    if (b) { c++; }
  }
  return c;
}

void Set::AddElement(std::size_t pos) {
  bits_.at(pos) = 1;
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

}

#endif
