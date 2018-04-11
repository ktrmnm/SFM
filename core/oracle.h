// Copyright 2018 Kentaro Minami. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef ORACLE_H
#define ORACLE_H

#include <vector>
#include <utility>
#include <memory>
#include <type_traits>
#include <string>
#include "core/utils.h"
#include "core/set_utils.h"
#include "core/partial_vector.h"
#include "core/reporter.h"

namespace submodular {

template <typename ValueType> class SubmodularOracle;
template <typename ValueType> class ReducibleOracle;

template <typename T>
struct ValueTraits{
  using value_type = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;
  //using rational_type = typename std::conditional<std::is_floating_point<T>::value, T, double>::type;
  using rational_type = double;
  using base_type = PartialVector<rational_type>;
};

template <typename ValueType>
class SubmodularOracle {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;

  virtual std::size_t GetN() const { return domain_.GetMembers().size(); }
  virtual std::size_t GetNGround() const { return domain_.n_; };
  virtual value_type Call(const Set& X) = 0;
  virtual std::string GetName() = 0;

  value_type Call(const Set& X, SFMReporter* reporter) {
    if (reporter != nullptr) {
      reporter->TimerStart(ReportKind::ORACLE);
    }
    auto ret = Call(X);
    if (reporter != nullptr) {
      reporter->TimerStop(ReportKind::ORACLE);
      reporter->IncreaseCount(ReportKind::ORACLE);
    }
    return ret;
  }

  Set GetDomain() const { return domain_.Copy(); }
  void SetDomain(const Set& X) { domain_ = X; }
  void SetDomain(Set&& X) { domain_ = std::move(X); }
  ReducibleOracle<ValueType> ToReducible() const { return ReducibleOracle<ValueType>(*this); }
protected:
  Set domain_;
};

template <typename ValueType>
class ReducibleOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;

  ReducibleOracle() = default; // todo: delete定義したい

  ReducibleOracle(const ReducibleOracle<ValueType>&) = default;
  ReducibleOracle(ReducibleOracle<ValueType>&&) = default;
  ReducibleOracle<ValueType>& operator=(const ReducibleOracle<ValueType>&) = default;
  ReducibleOracle<ValueType>& operator=(ReducibleOracle<ValueType>&&) = default;

  template <
    typename RawOracleType,
    std::enable_if_t<
      std::is_base_of<SubmodularOracle<ValueType>, RawOracleType>::value
      && !std::is_same<ReducibleOracle<ValueType>, RawOracleType>::value,
      std::nullptr_t
    > = nullptr
  >
  explicit ReducibleOracle(RawOracleType&& F)
    : contracted_(Set::MakeEmpty(F.GetNGround())),
      partition_(Partition::MakeFine(F.GetNGround())),
      offset_(value_type(0))
  {
    this->SetDomain(F.GetDomain());
    F_original_ = std::make_shared<RawOracleType>(std::move(F));
  };

  template <
    typename RawOracleType,
    std::enable_if_t<
      std::is_base_of<SubmodularOracle<ValueType>, RawOracleType>::value
      && !std::is_same<ReducibleOracle<ValueType>, RawOracleType>::value,
      std::nullptr_t
    > = nullptr
  >
  explicit ReducibleOracle(const RawOracleType& F)
    : contracted_(Set::MakeEmpty(F.GetNGround())),
      partition_(Partition::MakeFine(F.GetNGround())),
      offset_(value_type(0))
  {
    this->SetDomain(F.GetDomain());
    F_original_ = std::make_shared<RawOracleType>(F);
  };

  value_type Call(const Set& X);
  std::size_t GetN() const; // return the cardinality of the domain
  std::size_t GetNGround() const; // return the cardinality of the ground set

  std::string GetName() { return F_original_->GetName(); }

  // Make a reduction
  // The reduction (or restriction) F^A of a submodular function F to A is defined
  // as F^A(B) = F(B) for all subsets B of A.
  // This method simply shrink the original domain into A_new_domain.
  void Reduction(const Set& A_new_domain);
  ReducibleOracle<ValueType> ReductionCopy(const Set& A_new_domain) const;

  // Make a contraction
  // The contraction F_A of a submodular function F by A is defined as
  // F_A(B) = F(B \cup A) - F(A) for all subset B of the complement of A.
  // This method does (1) remove A_removed_from_domain from the original domain,
  // (2) add A_removed_from_domain to contracted_, and (3) add offset_ to the value of - F(A).
  void Contraction(const Set& A_removed_from_domain);
  ReducibleOracle<ValueType> ContractionCopy(const Set& A_removed_from_domain) const;

  // Make an oracle obtained by shrinking some nodes in the domain.
  // In the domain of the resulting oracle, the nodes in A_to_shrink is represented as a single node.
  void ShrinkNodes(const Set& A_to_shrink);
  ReducibleOracle<ValueType> ShrinkNodesCopy(const Set& A_to_shrink) const;

  value_type GetOffset() const { return offset_; }
  value_type SetOffset(value_type offset) { offset_ = offset; }
  void AddOffset(value_type offset) { offset_ += offset; }

protected:
  std::shared_ptr<SubmodularOracle<ValueType>> F_original_;
  Set contracted_;
  Partition partition_;
  value_type offset_;
};

template <typename ValueType>
typename ReducibleOracle<ValueType>::value_type
ReducibleOracle<ValueType>::Call(const Set& X) {
  // NOTE: If input set X contains any elements that are not contained in domain_,
  // they are just ignored.
  auto Y = X.Intersection(this->domain_).Union(contracted_);
  auto expanded = partition_.Expand(Y); // expand shrinked nodes
  return F_original_->Call(expanded) + offset_;
}

template <typename ValueType>
std::size_t ReducibleOracle<ValueType>::GetN() const {
  return this->domain_.Cardinality();
}

template <typename ValueType>
std::size_t ReducibleOracle<ValueType>::GetNGround() const {
  return F_original_->GetNGround();
}

template <typename ValueType>
void ReducibleOracle<ValueType>::Reduction(const Set& A_new_domain) {
  auto new_domain = this->GetDomain().Intersection(A_new_domain);
  this->SetDomain(std::move(new_domain));
}

template <typename ValueType>
ReducibleOracle<ValueType> ReducibleOracle<ValueType>::ReductionCopy(const Set& A_new_domain) const {
  ReducibleOracle<ValueType> reduction(*this);
  reduction.Reduction(A_new_domain);
  return reduction;
}

template <typename ValueType>
void ReducibleOracle<ValueType>::Contraction(const Set& A_removed_from_domain) {
  auto F_A = Call(A_removed_from_domain);
  AddOffset(-F_A);
  auto new_domain = this->GetDomain().Intersection(A_removed_from_domain.Complement());
  this->SetDomain(std::move(new_domain));
  auto new_contracted = contracted_.Union(A_removed_from_domain);
  contracted_ = std::move(new_contracted);
}

template <typename ValueType>
ReducibleOracle<ValueType> ReducibleOracle<ValueType>::ContractionCopy(const Set& A_removed_from_domain) const {
  ReducibleOracle<ValueType> contraction(*this);
  contraction.Contraction(A_removed_from_domain);
  return contraction;
}

template <typename ValueType>
void ReducibleOracle<ValueType>::ShrinkNodes(const Set& A_to_shrink) {
  auto members = A_to_shrink.GetMembers();
  auto new_index = partition_.MergeCells(members);

  // NOTE: if new_index == this->GetNGround(), there is no valid cell index in members.
  if (new_index < this->GetNGround()) {
    for (const auto& i: members) {
      if (i != new_index) {
        this->domain_.RemoveElement(i);
      }
    }
  }
}

template <typename ValueType>
ReducibleOracle<ValueType> ReducibleOracle<ValueType>::ShrinkNodesCopy(const Set& A_to_shrink) const {
  ReducibleOracle<ValueType> shrunk(*this);
  shrunk.ShrinkNodes(A_to_shrink);
  return shrunk;
}

/*
template <typename OracleType>
struct OracleTraits {
  using value_type = typename OracleType::value_type;
  using rational_type = typename OracleType::rational_type;
  using base_type = PartialVector<rational_type>;
};
*/

}// namespace submodular

#endif
