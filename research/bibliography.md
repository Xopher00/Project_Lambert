# Bibliography

The following is the full list of references has been collected during the course of research. These papers span several disciplines — mathematics, computer science, philosophy — that rarely cite one another. The central observation motivating this research is that fuzzy logic, formal concept analysis, associative memory, and modern neural attention mechanisms are all doing similar computations, expressed in different notations and developed by communities unaware of each other. This bibliography documents those parallels.

## Mathematic foundations

### Unified Algebra & Tensor Logic
Eric Hehner's Unified Algebra was the original motivation for this project — it shows that logical reasoning and arithmetic are special cases of the same algebra, not separate things. Pedro Domingos reaches a parallel conclusion from a completely different direction: starting from modern AI, he shows that neural networks and symbolic rule-based systems (like Datalog) use the same underlying tensor equations. These two threads, one from pure logic and one from machine learning, both point toward the same underlying structure that this project builds on.

- Hehner, E. C. R. (2004). From boolean algebra to unified algebra. The Mathematical Intelligencer, 26(2), 3–19. DOI: 10.1007/BF02985647.
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. International Journal of Mathematical Sciences, 1(1), 20–37. Available at: https://www.cs.toronto.edu/~hehner/UA.pdf
- Domingos, P. (2025). Tensor logic: The language of AI. arXiv preprint, arXiv:2510.12269.
- Shah, S., & Zadrozny, W. (2026). Implementing tensor logic: Unifying Datalog and neural reasoning via tensor contraction. arXiv preprint, arXiv:2601.17188.

### Fuzzy relations
Fuzzy logic replaces the binary true/false of classical logic with degrees of membership between 0 and 1. Zadeh's 1965 paper introduced this idea. Sanchez showed in 1976 how to solve relational equations under fuzzy composition. Dubois and Prade provide the broader theoretical context.

- Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338–353.
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. Information and Control, 30(1), 38–48.
- Dubois, D., & Prade, H. (1980). Fuzzy sets and systems: Theory and applications. Academic Press.

### Smooth approximation
Fuzzy max/min operations are not differentiable, which means they cannot be trained using standard gradient-based methods. Nesterov's work provides the mathematical basis for approximating these operations with smooth functions that can be. This is how Lambert bridges exact fuzzy logic and standard machine learning training.

- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. Mathematical Programming, Series A, 103, 127–152.

### Category theory
Category theory is a branch of mathematics that studies structure and relationships at a high level of abstraction. It provides the formal language for explaining why the parallels documented in this bibliography are not coincidences — why fuzzy logic, concept lattices, and neural attention share the same deep structure. 

- Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications. Pacific Journal of Mathematics, 5(2), 285–309.
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories. *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166. DOI: 10.1007/BF02924844. Republished as: *Reprints in Theory and Applications of Categories*, No. 1 (2002), pp. 1–37. Free PDF: http://www.tac.mta.ca/tac/reprints/articles/1/tr1abs.html
- Kelly, G. M. (1982). Basic concepts of enriched category theory. Cambridge University Press, Lecture Notes in Mathematics 64. Republished as: *Reprints in Theory and Applications of Categories*, No. 10 (2005). Free PDF: http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html
- Riehl, E. (2016). *Category theory in context*. Dover Publications. (Aurora: Dover Modern Math Originals series.) ISBN-13: 978-0-486-80903-8. 272 pp.
- Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via quantale-enriched two-variable adjunctions. Applied Categorical Structures, 29, 823–858. DOI: 10.1007/s10485-021-09654-w.

### Historical and philosophical context 
Burch's 1964 paper examines a seven-valued logic system from Jain philosophy, which has some interesting parallels with modern category theory. Grzejdziak-Zdziarski and Loog's 2026 report documents how the mathematical traditions represented in this bibliography developed in isolation from each other — which is itself part of the argument for why this project matters.

- Burch, G. B. (1964). Seven-valued logic in Jain philosophy. International Philosophical Quarterly, 4(1), 68–93.
- Grzejdziak-Zdziarski, M., & Loog, M. (2026). The missing interdiscipline: Reasons and ways to study the history of machine learning. External research report. Available at: https://hdl.handle.net/2066/328381.

## Technical core

### Fuzzy formal concept analysis - theory
Belohlavek spent decades developing the mathematics of how to find natural groupings in data using fuzzy logic. This section contains his core theoretical papers, along with Brito et al.'s accessible introduction to the field.

- Belohlavek, R. (c. 1998). Feedforward networks with fuzzy signals. Unpublished technical report, Institute for Research and Applications of Fuzzy Modeling / Department of Computer Science, Technical University of Ostrava. Available at: http://belohlavek.inf.upol.cz/publications/Bel_Fnfs.pdf
- Belohlavek, R. (2000). Fuzzy logical bidirectional associative memory. Information Sciences, 128, 91–103.
- Bělohlávek, R., & Vychodil, V. (2007). Fuzzy concept lattices constrained by hedges. Journal of Advanced Computational Intelligence and Intelligent Informatics, 11(6), 536–545. Publisher: Fuji Technology Press Ltd. ISSN: 1343-0130. URL: https://www.fujipress.jp/jaciii/jc/jacii001100060536/
- Belohlavek, R., & Vychodil, V. (2009). Formal concept analysis with background knowledge: Attribute priorities. IEEE Transactions on Systems, Man, and Cybernetics—Part C: Applications and Reviews, 39(4), 399–409.
- Brito, A. M., de Barros, L. C., Laureano, E. E., Bertato, F. M., & Coniglio, M. E. (2018). Fuzzy formal concept analysis. In Fuzzy Information Processing (pp. 192–205). Communications in Computer and Information Science. Springer. DOI: 10.1007/978-3-319-95312-0_17.

### Fuzzy FCA - algorithms
These papers ask how to find natural groupings in data efficiently, rather than what those groupings are. The GreConD algorithm (Belohlavek & Vychodil 2010) reframes the problem as matrix decomposition. Trnecka revisits and benchmarks this approach. Kriegel and Borchmann's NextClosures is the standard alternative, included here as a comparison point.

- Belohlavek, R., & Vychodil, V. (2010). Discovery of optimal factors in binary data via a novel method of matrix decomposition. Journal of Computer and System Sciences, 76(1), 3–20. DOI: 10.1016/j.jcss.2009.05.002.
- Belohlavek, R., & Trnecka, M. (2015). From-below approximations in Boolean matrix factorization: Geometry and new algorithm. Journal of Computer and System Sciences, 81(8), 1678–1697. DOI: 10.1016/j.jcss.2015.06.002.
- Kriegel, F., & Borchmann, D. (2015). NextClosures: Parallel computation of the canonical base. In Proceedings of the 12th International Conference on Concept Lattices and Their Applications (CLA 2015). TU Dresden.
- Trnecka, M., & Vyjidacek, R. (2020). Revisiting the GreCon algorithm for boolean matrix factorization. In F. J. Valverde-Albacete & M. Trnecka (Eds.), Proceedings of the 15th International Conference on Concept Lattices and Their Applications (CLA 2020), CEUR Workshop Proceedings, vol. 2668, pp. 59–70.

### FCA extended
These push the core FCA framework into less standard territory. Belohlavek, Krmelova and Outrata (2010) address how to compute the full set of stable points of a fuzzy closure operator. Bazin et al. (2024) extend the framework from two-way object-attribute relations to relations involving three or more dimensions. Alcântara et al. (2021) is a survey situating fuzzy relations within a broader mathematical context.

- Belohlavek, R., Krmelova, M., & Outrata, J. (2010). Computing the lattice of all fixpoints of a fuzzy closure operator. IEEE Transactions on Fuzzy Systems, 18(3), 546–557.
- Alcântara, M. S. da S., Dias, T., de Oliveira, W. R., & de Melo, S. de B. (2021). A survey of categorical properties of L-fuzzy relations. Fuzzy Sets and Systems. DOI: 10.1016/j.fss.2021.03.010.
- Bazin, A., Galasso-Carbonnel, J., & Kahn, G. (2024). Polyadic relational concept analysis. International Journal of Approximate Reasoning, 164, 109067. DOI: 10.1016/j.ijar.2023.109067.
- Schmitt, I. (2026). Triadic concept analysis for logic interpretation of simple artificial networks. arXiv preprint, arXiv:2601.06229.

### Morphological and associative memory
These are the direct historical predecessors of the attention mechanism used in this project. Ritter, Sussner and Díaz-de-León showed in 1998–1999 that neural networks built on max and min operations — rather than multiply-and-add — can function as associative memories: given a partial or noisy pattern, they retrieve the closest stored one. Sussner and Valle (2006) extended this using fuzzy implication as the retrieval operation.

- Ritter, G. X., Sussner, P., & Díaz-de-León, J. L. (1998). Morphological associative memories. IEEE Transactions on Neural Networks, 9(2), 281–293. DOI: 10.1109/72.661123.
- Ritter, G. X., Díaz-de-León, J. L., & Sussner, P. (1999). Morphological bidirectional associative memories. Neural Networks, 12(6), 851–867. DOI: 10.1016/S0893-6080(99)00033-7.
- Sussner, P., & Valle, M. E. (2006). Implicative fuzzy associative memories. IEEE Transactions on Fuzzy Systems, 14(6), 791–807.

###  Modern Hopfield and attention
Transformer attention and Hopfield network memory retrieval are mathematically equivalent — this was formally shown in 2021 by two independent papers arriving at the same result from different directions. Krotov and Hopfield derive it from the neuroscience side; Ramsauer et al. from the engineering side. Both are included because they frame the same equivalence differently, and together they connect the earlier morphological memory literature to the modern transformer architecture.

- Krotov, D., & Hopfield, J. (2021). Large associative memory problem in neurobiology and machine learning. In Proceedings of the International Conference on Learning Representations (ICLR 2021). arXiv:2008.06996.
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2021). Hopfield networks is all you need. In Proceedings of the International Conference on Learning Representations (ICLR 2021). arXiv:2008.02217.

### Semiring provenance and fixpoints
When a query returns an answer, it is useful to know why — which facts and relationships produced it. This is called provenance. Green, Karvounarakis and Tannen (2007) developed the algebraic framework for tracking this through database queries. The later papers extend it to iterative computation and prove properties of the specific algebraic structure used in this project.

- Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings. In Proceedings of the 26th ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems (PODS 2007), pp. 31–40. DOI: 10.1145/1265530.1265535.
- Dannert, K. M., Grädel, E., Naaf, M., & Tannen, V. (2021). Semiring provenance for fixed-point logic. In Proceedings of the 29th EACSL Annual Conference on Computer Science Logic (CSL 2021), LIPIcs vol. 183, article 17. DOI: 10.4230/LIPIcs.CSL.2021.17.
- Bizière, C., Grädel, E., & Naaf, M. (2023). Locality theorems in semiring semantics. In *Proceedings of the 48th International Symposium on Mathematical Foundations of Computer Science (MFCS 2023)* (J. Leroux, S. Lombardy, & D. Peleg, Eds.), vol. 272 of LIPIcs, pp. 20:1–20:15. Schloss Dagstuhl – Leibniz-Zentrum für Informatik. DOI: 10.4230/LIPIcs.MFCS.2023.20
- Naaf, M. (2024). Logic, semirings, and fixed points, PhD thesis, RWTH Aachen University. Defended 30 August 2024. DOI: 10.18154/RWTH-2024-10804. URL: https://publications.rwth-aachen.de/record/996756/files/996756.pdf

### Predictive coding / energy
Iterative computation of the kind used in this project can be understood as minimising an energy function — each step brings the current state closer to a stable answer. Friston, Parr and Pezzulo (2022) develop this framing in the context of biological perception, where the same mathematics describes how the brain updates its model of the world given new evidence. Qi et al. and Burchi and Timofte bring this into contact with modern neural network training.

- Friston, K. J., Parr, T., & Pezzulo, G. (2022). Active inference: The free energy principle in mind, brain, and behavior. MIT Press.
- Qi, C., Lukasiewicz, T., & Salvatori, T. (2025). Training deep predictive coding networks. In New Frontiers in Associative Memory Workshop, ICLR 2025.
- Burchi, M., & Timofte, R. (2025). Learning transformer-based world models with contrastive predictive coding. International Conference on Learning Representations (ICLR 2025).

## Landscape and positioning

### Categorical deep learning
Category theory is increasingly being applied to understand deep learning architectures from first principles rather than just describing them empirically. These papers represent that direction. Gavranović et al.'s 2024 ICML paper argues that all standard neural architectures are instances of a single categorical framework — where this project sits relative to that claim is a live theoretical question. Manin and Marcolli apply the same categorical language specifically to Hopfield-style memory dynamics.

- Jones, I., Swan, J., & Giansiracusa, J. (2024). Algebraic dynamical systems in machine learning. Applied Categorical Structures, 32, article 4. DOI: 10.1007/s10485-023-09762-9.
- Gavranović, B. (2024). Fundamental Components of Deep Learning: A category-theoretic approach. PhD thesis, University of Strathclyde. arXiv:2403.13001.
- Gavranović, B., Lessard, P., Dudzik, A., von Glehn, T., Araújo, J.G.M. & Veličković, P. (2024). Position: Categorical deep learning is an algebraic theory of all architectures. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).
- Manin, Y. I., & Marcolli, M. (2024). Homotopy theoretic and categorical models of neural information networks. Compositionality, 6(4), 14135. DOI: 10.46298/compositionality-6-4.

### Tropical and morphological geometry
Tropical geometry replaces the standard operations of arithmetic with max and addition, producing a piecewise-linear framework that has been used to characterise the decision boundaries of ReLU networks. This project uses max and min instead — a different algebra with different properties. These papers document the adjacent tropical geometry literature, which helps clarify what makes the two approaches distinct.

- Maragos, P. (2005). Lattice image processing: A unification of morphological and fuzzy algebraic systems. Journal of Mathematical Imaging and Vision, 22, 83–118.
- Zhang, L., Naitzat, G., & Lim, L.-H. (2018). Tropical geometry of deep neural networks. In Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80. arXiv:1805.07091.
- Alfarra, M. H. A. (2020). Applications of tropical geometry in deep neural networks. MSc Thesis, King Abdullah University of Science and Technology (KAUST).
- Maragos, P., Charisopoulos, V., & Theodosis, E. (2021). Tropical Geometry and Machine Learning. Proceedings of the IEEE, 109(5), 2073–2088. DOI: 10.1109/JPROC.2021.3065238.
- Alfarra, M., Bibi, A., Hammoud, H., Gaafar, M., & Ghanem, B. (2023). On the decision boundaries of neural networks: A tropical geometry perspective. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4), 5027–5037. DOI: 10.1109/TPAMI.2022.3201490.

### Neurosymbolic reasoning
In recent years there has been a lot of research into neurosymbolic AI, especially as the limits of LLMs become more apparent. This work sits alongside Domingos in the literature.

- Serafini, L., d'Avila Garcez, A., Badreddine, S., Donadello, I., Spranger, M., & Bianchi, F. (2022). Logic tensor networks: Theory and applications. In P. Hitzler & M. K. Sarker (Eds.), Neuro-Symbolic Artificial Intelligence: The State of the Art (Ch. 17). IOS Press.
- Goessmann, A., Schütte, J., Fröhlich, M., & Eigel, M. (2026). A tensor network formalism for neuro-symbolic AI. arXiv preprint, arXiv:2601.15442.
- Ren W., Wan K., Leng J., & Li S. (2026). Inferring the Invisible: Neuro-Symbolic Rule Discovery for Missing Value Imputation. ICLR 2026. URL: https://openreview.net/forum?id=26Msp6pV5i

### Interpretability
These papers address the question of whether an AI system can show its reasoning — tracing an answer back to the specific facts that produced it, rather than producing an answer with no explanation. Wattenberg and Viégas (2024) specifically identify relational composition as an unsolved interpretability problem. Schmitt (2026) applies formal concept analysis directly to interpreting neural networks, making it a close parallel to work in the technical core section above.

- Wattenberg, M., & Viégas, F. B. (2024). Relational composition in neural networks: A survey and call to action. arXiv preprint, arXiv:2407.14662.
- Tan, X. W., Tan, N., Lee, G., & Kok, S. (2025). The shape of reasoning: Topological analysis of reasoning traces in large language models. arXiv preprint, arXiv:2510.20665.
- Hamilton, A., Wright, E. P., & Vance, C. (2026). Integrating symbolic reasoning into neural networks: A neuro-symbolic logic programming approach for enhanced explainability. Frontiers in Artificial Intelligence Research, 3(1), 54–62. DOI: 10.71465/fair602.
