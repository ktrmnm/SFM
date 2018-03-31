# [WIP] Submodular Function Minimization

under construction

## TODO

Algorithms

- [x] Robust Fujishige--Wolfe [Fujishige1984] [Chakrabarty+2014]
- [ ] Iwata--Fleischer--Fujishige's algorithm (weakly polynomial) [Iwata+2001]
- [ ] Iwata--Fleischer--Fujishige's algorithm (strongly polynomial) [Iwata+2001]
- [ ] Iwata--Orlin's algorithm [Iwata&Orlin2009]
- [ ] Schrijver's algorithm [Schrijver2000]
- [ ] Cutting plane method proposed in [Lee+2015]
- [x] Generalized graph cut function minimization [Jegelka+2011]
- [x] Brute force
- [x] Divide-conquer algorithm for minimum norm point

## References

```
[Chakrabarty+2014] Chakrabarty, Jain, Kothari (2014). Provable submodular minimization using Wolfe’s algorithm. In NIPS.
[Fujishige1984] Fujishige (1884). Submodular systems and related topics. Math. Programming Study.
[Iwata+2001] Iwata, Fleischer, Fujishige (2001). A combinatorial strongly polynomial algorithm for minimizing submodular functions. Journal of the ACM (JACM), 48(4):761– 777.
[Iwata&Orlin2009] Iwata, Orlin (2009). A simple combinatorial algorithm for submodular function minimization. In SODA.
[Jegelka+2011] Jegelka, Lin, Blimes (2011). On fast approximate submodular minimization. In NIPS.
[Lee+2015] Lee, Sidford, Wong (2015). A faster cutting plane method and its implications for combinatorial and convex optimization. In FOCS.
[Schrijver2000] Schrijver (2000). A combinatorial algorithm minimizing submodular functions in strongly polynomial time.
J. Comb. Theory, Ser. B, 80(2):346–355.
```

## License

- Apache License 2.0 (see LICENSE)
- The linear algebra module uses a part of [Eigen](http://eigen.tuxfamily.org/) v3.3.4 which is licensed under the Mozilla Public License Version 2.0.
