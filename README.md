### 🔹 Частина 1 — Підготовка даних і вибір інструментів
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\01_prepare_data.py        
Читаємо датасет: 10000it [00:00, 39895.07it/s]
\nЗавантажено статей:10000
\nРозподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64
\nРозподіл за роками:
year
2007    10000
Name: count, dtype: int64
\nПриклад запису:
{'id': '0704.0001', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies', 'abstract': 'A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.', 'authors': 'BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.', 'year': 2007, 'category': 'hep-ph'}
\nЗбережено вdata/arxiv_subset.parquet
```

<br>

1.2.

<br>

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

Pinecone відрізняється від Qdrant і Chroma тим, що:
- має пропрієтарну ліцензію(код закритий)
- це суто хмарний сервіс, який сам бере на себе процес розгортання інфраструктури, шардінг, масштабування тощо. Клієнт про все це може не турбуватися.

Chroma краще обирати для локальних експериментів та прототипування. Для потужного власного бекенду в контейнерах — Qdrant. Для хмарного продакшену, де інфраструктурою керує хтось інший — Pinecone.

<br>

2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

Тому що модель specter2_base була натренована на наукових текстах, що дозволяє використовувати її для задач класифікації, регресії, знаходження схожості статей, використання в пошукових системах. 

"SPECTER2 has been trained on over 6M triplets of scientific paper citations, which are available here. Post that it is trained with additionally attached task format specific adapter modules on all the SciRepEval training tasks.
Task Formats trained on:
    Classification
    Regression
    Proximity (Retrieval)
    Adhoc Search
"

<br>

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?
У картці моделі я щось нічого не знайшла про метрику схожості, але якщо вірити ШІ, то рекомендованою метрикою є косинусна схожість. 
При створені індексу система (напр. Pinecone) жорстко вимагає вказати метрику відстані. 

<br>

1.3
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\02_embed.py       
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [13:49<00:00,  5.28s/it]

--- Результати ---
Загальна кількість оброблених текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.0000
```

Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

Якщо додати ембедінги одиничної довжини в формулу косинусної схожості - вона перетворюється в формулу:
$Cosine(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = Cosine(A, B) = \frac{A \cdot B}{1 \cdot 1} = A \cdot B$

А це дорівнює скалярному добутку:
$Dot Product(A, B) = A \cdot B$

<br>

### 🔹 Частина 2 — Завантаження даних і метадані
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\03_load_to_pinecone.py
          id                                              title  ...  year        category
0  0704.0001  Calculation of prompt diphoton production cros...  ...  2007          hep-ph
1  0704.0002           Sparsity-certifying Graph Decompositions  ...  2007         math.CO
2  0704.0003  The evolution of the Earth-Moon system based o...  ...  2007  physics.gen-ph
3  0704.0004  A determinant of Stirling cycle numbers counts...  ...  2007         math.CO
4  0704.0005  From dyadic $\Lambda_{\alpha}$ to $\Lambda_{\a...  ...  2007         math.CA

[5 rows x 6 columns]
Завантаження батчів: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:41<00:00,  1.21it/s]
****************************************
Загальна кількість векторів в індексі: 10000
****************************************
```

<br>

### 🔹 Частина 3 — Пошукові запити
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\04_search.py          
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Запит: 'teaching machines to recognize objects in pictures':
============================================================
1. [Схожість: 0.8288] Capturing knots in polymers
   Категорія: cond-mat.soft | Рік публікації: 2007.0
   Анотація (уривок): This paper visualizes a knot reduction algorithm...
------------------------------------------------------------
2. [Схожість: 0.8263] Symbolic sensors : one solution to the numerical-symbolic interface
   Категорія: physics.ins-det | Рік публікації: 2007.0
   Анотація (уривок): This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, the links between the physical world and the symbolic
one are introduced. The creation of symbols is proposed within the frame of the
pretopology theo...
------------------------------------------------------------
3. [Схожість: 0.8256] The Mathematics
   Категорія: math.HO | Рік публікації: 2007.0
   Анотація (уривок): This is an essay that considering the knowledge structure and language of a
different nature, attempts to build on an explanation of the object of study
and characteristics of the mathematical science. We end up with a learning
cycle of mathematics a...
------------------------------------------------------------
4. [Схожість: 0.8170] Modeling the field of laser welding melt pool by RBFNN
   Категорія: physics.comp-ph | Рік публікації: 2007.0
   Анотація (уривок): Efficient control of a laser welding process requires the reliable prediction
of process behavior. A statistical method of field modeling, based on
normalized RBFNN, can be successfully used to predict the spatiotemporal
dynamics of surface optical a...
------------------------------------------------------------
5. [Схожість: 0.8146] Why should anyone care about computing with anyons?
   Категорія: quant-ph | Рік публікації: 2007.0
   Анотація (уривок): In this article we present a pedagogical introduction of the main ideas and
recent advances in the area of topological quantum computation. We give an
overview of the concept of anyons and their exotic statistics, present various
models that exhibit ...
------------------------------------------------------------
 За цими фільтрами нічого не знайдено.
Запит: 'reinforcement learning':
============================================================
1. [Схожість: 0.8445] Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa
   Категорія: cs.MA | Рік публікації: 2007.0
   Анотація (уривок): Game theory has many limitations implicit in its application. By utilizing
multiagent modeling, it is possible to solve a number of problems that are
unsolvable using traditional game theory. In this paper reinforcement learning
is applied to neural ...
------------------------------------------------------------
2. [Схожість: 0.8194] Introduction to Phase Transitions in Random Optimization Problems
   Категорія: cond-mat.stat-mech | Рік публікації: 2007.0
   Анотація (уривок): Notes of the lectures delivered in Les Houches during the Summer School on
Complex Systems (July 2006)....
------------------------------------------------------------
3. [Схожість: 0.8102] Architecture for Pseudo Acausal Evolvable Embedded Systems
   Категорія: cs.NE | Рік публікації: 2007.0
   Анотація (уривок): Advances in semiconductor technology are contributing to the increasing
complexity in the design of embedded systems. Architectures with novel
techniques such as evolvable nature and autonomous behavior have engrossed lot
of attention. This paper dem...
------------------------------------------------------------
4. [Схожість: 0.8010] Why only few are so successful ?
   Категорія: physics.pop-ph | Рік публікації: 2007.0
   Анотація (уривок): In many professons employees are rewarded according to their relative
performance. Corresponding economy can be modeled by taking $N$ independent
agents who gain from the market with a rate which depends on their current
gain. We argue that this simp...
------------------------------------------------------------
5. [Схожість: 0.7993] Opinion Dynamics and Sociophysics
   Категорія: physics.soc-ph | Рік публікації: 2007.0
   Анотація (уривок): No abstract given. Contents:
  I. Definition and Introduction
  II. Schelling Model
  III. Opinion Dynamics
  IV. Languages, Hierarchies and Football
  V. Future Directions...
------------------------------------------------------------
Метрика: 1. Cosine Similarity
1. [Score/Dist: 0.8294] Capturing knots in polymers
2. [Score/Dist: 0.8260] Symbolic sensors : one solution to the numerical-symbolic interface
3. [Score/Dist: 0.8254] The Mathematics
4. [Score/Dist: 0.8181] Modeling the field of laser welding melt pool by RBFNN
5. [Score/Dist: 0.8142] Python for Education: Computational Methods for Nonlinear Systems
Метрика: 2. Dot Product
1. [Score/Dist: 0.8294] Capturing knots in polymers
2. [Score/Dist: 0.8260] Symbolic sensors : one solution to the numerical-symbolic interface
3. [Score/Dist: 0.8254] The Mathematics
4. [Score/Dist: 0.8181] Modeling the field of laser welding melt pool by RBFNN
5. [Score/Dist: 0.8142] Python for Education: Computational Methods for Nonlinear Systems
Метрика: 3. L2 Distance (Менше - краще)
1. [Score/Dist: 0.5842] Capturing knots in polymers
2. [Score/Dist: 0.5899] Symbolic sensors : one solution to the numerical-symbolic interface
3. [Score/Dist: 0.5910] The Mathematics
4. [Score/Dist: 0.6032] Modeling the field of laser welding melt pool by RBFNN
5. [Score/Dist: 0.6095] Python for Education: Computational Methods for Nonlinear Systems
```

<br>

В результаті виконання запиту фільрації запрос по категорії cs.LG за остані 5 років не знайшов жодної статті. Це можна пояснити тим, як була опрацьована категорія в скрипті "01_prepare_data.py"(код брався із завдання). В оригінальному json файлі багато статей які мають декілька категорій. При обробці цих категорій береться лише перша з них,а інші відкидаються. Через це багато статей категорії cs.LG були відфільтровані,а на виході ми отримали лише 10 статей категорії cs.LG і всі вони 2007 року. Тобто жодна з них не підходить під наш запит.
 
<br>

1.  Чи збігаються топ-5 для cosine і dot product і чому?
Так, тому що вектори були попередньо нормалізовані, а для нормалізованих векторів метрики Cosine similarity і Dot Product - це одне і те саме.

<br>

2. Чи відрізняються результати для L2 і чому?
Відрізняються лише значенням score, набір і порядок статей залишається той самий - це пов'язано з тим, що для нормальнізованих векторів існує математичний зв’язок між L2 та косинусною схожістю. Це означає, що на нормалізованих векторах обидві метрики дадуть однаковий порядок ранжування результатів — вектор із мінімальною L2-відстанню також матиме максимальну косинусну схожість.

<br>

3. Що сталося б, якби ембеддинги не були нормалізовані?
Якби вектори не були нормалізовані результати ранжування для цих трьох метрик сильно відрізнялися б: косинусна схожість залишилась б актуальною, а от Dot Product і L2 - працювали б некоректно.

<br>

### 🔹 Частина 4 — Chunking
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\05_chunking.py
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Індекс 'arxiv-chunks-fixed' вже існує.
Індекс 'arxiv-chunks-semantic' вже існує.
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.16s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.90s/it]
Завантаження завершено. Векторів в індексі: 413
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.58s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.69s/it]
Завантаження завершено. Векторів в індексі: 30
--------------------------------------------------------------------------------
Запит: 'quantum entanglement in complex systems'
--------------------------------------------------------------------------------
--- Результати з індексу: FIXED SIZE CHUNKING ---
1. [Схожість: 0.8217] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #2.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "dynamical mapping (or recursive lattices) for investigation of magnetic
properties of the fluid and solid $^3$He, phase transitions in crystals and"
------------------------------------------------------------
2. [Схожість: 0.8211] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #0.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "In this thesis, we consider some spin effects in QCD and recurrence lattices
with multi-site exchanges. Main topic of our manuscript are critical phenomena
in spin systems defined on the recurrence lattices. Main tool of our approach
is the method of recursive (hierarchical) lattices. We apply the method of
dynamical mapping (or recursive lattices) for investigation of magnetic
properties of the fluid and solid $^3$He, phase transitions in crystals and
macromolecules. First, we analyze the helix-coil phase transition for
polypeptides and proteins, and describe an quasi unfolding transition (like the
cold denaturation process) for the degree of helicity (the order parameter for
macromolecules). Next we consider the recurrent models of $^3$He defined on the
square, Husimi and hexagon lattices. Using the method of dynamical mapping, the
magnetization curves with plateaus, bifurcation point and one period doubling
are obtained. Then we investigate the model with cubic symmetry defined on the
Bethe lattice and containing both linear and quadratic spin-spin interactions.
The magnetization of the system is calculated, and a complex structure of the
phase transitions between the disordered, partially ordered and completely
ordered states is observed. In the framework of QCD, we consider the azimuthal"
------------------------------------------------------------
3. [Схожість: 0.8191] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #4.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "cold denaturation process) for the degree of helicity (the order parameter for
macromolecules). Next we consider the recurrent models of $^3$He defined on the"
------------------------------------------------------------
4. [Схожість: 0.8186] Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
Чанк #2.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "condition to deal with the topologic problem of the 3D Ising model"
------------------------------------------------------------
5. [Схожість: 0.8151] Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
Чанк #3.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: ". The partition function of the 3D simple orthorhombic Ising model is evaluated by spinor analysis, by employing these conjectures"
------------------------------------------------------------
--- Результати з індексу: SEMANTIC CHUNKING ---
1. [Схожість: 0.8157] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #0.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in spin systems defined on the recurrence lattices. Main tool of our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical mapping (or recursive lattices) for investigation of magnetic properties of the fluid and solid $^3$He, phase transitions in crystals and macromolecules. First, we analyze the helix-coil phase transition for polypeptides and proteins, and describe an quasi unfolding transition (like the cold denaturation process) for the degree of helicity (the order parameter for macromolecules). Next we consider the recurrent models of $^3$He defined on the square, Husimi and hexagon lattices. Using the method of dynamical mapping, the magnetization curves with plateaus, bifurcation point and one period doubling are obtained. Then we investigate the model with cubic symmetry defined on the Bethe lattice and containing both linear and quadratic spin-spin interactions. The magnetization of the system is calculated, and a complex structure of the phase transitions between the disordered, partially ordered and completely ordered states is observed. In the framework of QCD, we consider the azimuthal asymmetries in heavy flavor production in the lepton-nucleon deep inelastic scattering (DIS). We calculate the azimuthal (or $\phi$-) dependence of the next-to-leading order heavy-quark-initiated contributions to DIS. It is shown that, contrary to the basic gluon-initiated component, the photon-quark scattering mechanism is practically $\cos2\phi$-independent. We investigate the possibility of measuring both nonperturbative (intrinsic) and perturbative (CTEQ, MRST) charm distributions using the $\cos2\phi$ asymmetry."
------------------------------------------------------------
2. [Схожість: 0.6950] Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
Чанк #0.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "We report the conjectures on the three-dimensional (3D) Ising model on simple orthorhombic lattices, together with the details of calculations for a putative exact solution. Two conjectures, an additional rotation in the fourth curled-up dimension and the weight factors on the eigenvectors, are proposed to serve as a boundary condition to deal with the topologic problem of the 3D Ising model. The partition function of the 3D simple orthorhombic Ising model is evaluated by spinor analysis, by employing these conjectures. Based on the validity of the conjectures, the critical temperature of the simple orthorhombic Ising lattices could be determined by the relation of KK* = KK' + KK'' + K'K'' or sinh 2K sinh 2(K' + K'' + K'K''/K) = 1. For a simple cubic Ising lattice, the critical point is putatively determined to locate exactly at the golden ratio xc = exp(-2Kc) = (sq(5) - 1)/2, as derived from K* = 3K or sinh 2K sinh 6K = 1. If the conjectures would be true, the specific heat of the simple orthorhombic Ising system would show a logarithmic singularity at the critical point of the phase transition. The spontaneous magnetization and the spin correlation functions of the simple orthorhombic Ising ferromagnet are derived explicitly. The putative critical exponents derived explicitly for the simple orthorhombic Ising lattices are alpha = 0, beta = 3/8, gamma = 5/4, delta = 13/3, eta = 1/8 and nu = 2/3, showing the universality behavior and satisfying the scaling laws. The cooperative phenomena near the critical point are studied and the results obtained based on the conjectures are compared with those of the approximation methods and the experimental findings. The 3D to 2D crossover phenomenon differs with the 2D to 1D crossover phenomenon and there is a gradual crossover of the exponents from the 3D values to the 2D ones."
------------------------------------------------------------
3. [Схожість: 0.6837] Spherically symmetric problem on the brane and galactic rotation curves
Чанк #0.0 (Рік: 2007.0, Категорія: gr-qc)
Текст: "We investigate the braneworld model with induced gravity to clarify the role of the cross-over length scale \ell in the possible explanation of the dark-matter phenomenon in astrophysics and in cosmology. Observations of the 21 cm line from neutral hydrogen clouds in spiral galaxies reveal that the rotational velocities remain nearly constant at a value v_c ~ 10^{-3}--10^{-4} in the units of the speed of light in the region of the galactic halo. Using the smallness of v_c, we develop a perturbative scheme for reconstructing the metric in a galactic halo. In the leading order of expansion in v_c, at the distances r \gtrsim v_c \ell, our result reproduces that obtained in the Randall-Sundrum braneworld model. This inequality is satisfied in a real spiral galaxy such as our Milky Way for distances r ~ 3 kpc, at which the rotational velocity curve becomes flat, v_c ~ 7 \times 10^{-4}, if \ell \lesssim 2 Mpc. The gravitational situation in this case can be approximately described by the Einstein equations with the so-called Weyl fluid playing the role of dark matter. In the region near the gravitating body, we derive a closed system of equations for static spherically symmetric situation under the approximation of zero anisotropic stress of the Weyl fluid. We find the Schwarzschild metric to be an approximate vacuum solution of these equations at distances r \lesssim (r_g \ell^2)^{1/3}. The value \ell \lesssim 2 Mpc complies well with the solar-system tests. At the same time, in cosmology, a low-density braneworld with \ell of this order of magnitude can mimic the expansion properties of the high-density LCDM (lambda + cold dark matter) universe at late times. Combined observations of galactic rotation curves and gravitational lensing can possibly discriminate between the higher-dimensional effects and dark matter."
------------------------------------------------------------
4. [Схожість: 0.6727] Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We here develop an improved way of using a rotating star as a clock, set it using the Sun, and demonstrate that it keeps time well. This technique, called gyrochronology, permits the derivation of ages for solar- and late-type main sequence stars using only their rotation periods and colors. The technique is clarified and developed here, and used to derive ages for illustrative groups of nearby, late-type field stars with measured rotation periods. We first demonstrate the reality of the interface sequence, the unifying feature of the rotational observations of cluster and field stars that makes the technique possible, and extends it beyond the proposal of Skumanich by specifying the mass dependence of rotation for these stars. We delineate which stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their dependence on color and period. Representative age errors associated with the technique are estimated at ~15% (plus possible systematic errors) for late-F, G, K, & early-M stars. Ages derived via gyrochronology for the Mt. Wilson stars are shown to be in good agreement with chromospheric ages for all but the bluest stars, and probably superior. Gyro ages are then calculated for each of the active main sequence field stars studied by Strassmeier and collaborators where other ages are not available. These are shown to be mostly younger than 1Gyr, with a median age of 365Myr. The sample of single, late-type main sequence field stars assembled by Pizzolato and collaborators is then assessed, and shown to have gyro ages ranging from under 100Myr to several Gyr, and a median age of 1. 2Gyr. Finally, we demonstrate that the individual components of the three wide binaries XiBooAB, 61CygAB, & AlphaCenAB yield substantially the same gyro ages."
------------------------------------------------------------
5. [Схожість: 0.6724] Dependence of CMI Growth Rates on Electron Velocity Distributions and
  Perturbation by Solitary Waves
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We calculate growth rates and corresponding gains for RX and LO mode radiation associated with the cyclotron maser instability for parameterized horseshoe electron velocity distributions. The velocity distribution function was modeled to closely fit the electron distribution functions observed in the auroral cavity. We systematically varied the model parameters as well as the propagation direction to study the dependence of growth rates on model parameters. The growth rate depends strongly on loss cone opening angle, which must be less than $90^{o}$ for significant CMI growth. The growth rate is sharply peaked for perpendicular radiation ($k_{\parallel} = 0$), with a full-width at half-maximum $1. 7^{o}$, in good agreement with observed k-vector orientations and numerical simulations. The fractional bandwidth varied between 10$^{-4}$ and 10$^{-2}$, depending most strongly on propagation direction. This range encompasses nearly all observed fractional AKR burst bandwidths. We find excellent agreement between the computed RX mode emergent intensities and observed AKR intensities assuming convective growth length $L_c\approx$20-40 km and group speed 0. 15$c$. The only computed LO mode growth rates compatible observed LO mode radiation levels occurred for number densities more than 100 times the average energetic electron densities measured in auroral cavities. This implies that LO mode radiation is not produced directly by the CMI mechanism but more likely results from mode conversion of RX mode radiation. We find that perturbation of the model velocity distribution by large ion solitary waves (ion holes) can enhance the growth rate by a factor of 2-4. This will result in a gain enhancement more than 40 dB depending on the convective growth length within the structure. Similar enhancements may be caused by EMIC waves."
------------------------------------------------------------
--------------------------------------------------------------------------------
Запит: 'deep learning architectures for natural language'
--------------------------------------------------------------------------------
--- Результати з індексу: FIXED SIZE CHUNKING ---
1. [Схожість: 0.7858] Our Peculiar Motion Away from the Local Void
Чанк #13.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". The Local Void is large!"
------------------------------------------------------------
2. [Схожість: 0.7784] Rotation and activity of pre-main-sequence stars
Чанк #12.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". (abridged)"
------------------------------------------------------------
3. [Схожість: 0.7784] Improved constraints on dark energy from Chandra X-ray observations of
  the largest relaxed galaxy clusters
Чанк #13.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". (Abridged)"
------------------------------------------------------------
4. [Схожість: 0.7663] The Kinematics of the Ultra-Faint Milky Way Satellites: Solving the
  Missing Satellite Problem
Чанк #14.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". [slightly abridged]"
------------------------------------------------------------
5. [Схожість: 0.7627] Spherically symmetric problem on the brane and galactic rotation curves
Чанк #10.0 (Рік: 2007.0, Категорія: gr-qc)
Текст: "with \ell of this order of magnitude can mimic the expansion properties of the
high-density LCDM (lambda + cold dark matter) universe at late times. Combined"
------------------------------------------------------------
--- Результати з індексу: SEMANTIC CHUNKING ---
1. [Схожість: 0.7177] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #0.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in spin systems defined on the recurrence lattices. Main tool of our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical mapping (or recursive lattices) for investigation of magnetic properties of the fluid and solid $^3$He, phase transitions in crystals and macromolecules. First, we analyze the helix-coil phase transition for polypeptides and proteins, and describe an quasi unfolding transition (like the cold denaturation process) for the degree of helicity (the order parameter for macromolecules). Next we consider the recurrent models of $^3$He defined on the square, Husimi and hexagon lattices. Using the method of dynamical mapping, the magnetization curves with plateaus, bifurcation point and one period doubling are obtained. Then we investigate the model with cubic symmetry defined on the Bethe lattice and containing both linear and quadratic spin-spin interactions. The magnetization of the system is calculated, and a complex structure of the phase transitions between the disordered, partially ordered and completely ordered states is observed. In the framework of QCD, we consider the azimuthal asymmetries in heavy flavor production in the lepton-nucleon deep inelastic scattering (DIS). We calculate the azimuthal (or $\phi$-) dependence of the next-to-leading order heavy-quark-initiated contributions to DIS. It is shown that, contrary to the basic gluon-initiated component, the photon-quark scattering mechanism is practically $\cos2\phi$-independent. We investigate the possibility of measuring both nonperturbative (intrinsic) and perturbative (CTEQ, MRST) charm distributions using the $\cos2\phi$ asymmetry."
------------------------------------------------------------
2. [Схожість: 0.6708] Multicolor observations of the afterglow of the short/hard GRB 050724
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "New information on short/hard gamma-ray bursts (GRBs) is being gathered thanks to the discovery of their optical and X-ray afterglows. However, some key aspects are still poorly understood, including the collimation level of the outflow, the duration of the central engine activity, and the properties of the progenitor systems. We want to constrain the physical properties of the short GRB 050724 and of its host galaxy, and make some inferences on the global short GRB population. We present optical observations of the afterglow of GRB 050724 and of its host galaxy, significantly expanding the existing dataset for this event. We compare our results with models, complementing them with available measurements from the literature. We study the afterglow light curve and spectrum including X-ray data. We also present observations of the host galaxy. The observed optical emission was likely related to the large flare observed in the X-ray light curve. The apparent steep decay was therefore not due to the jet effect. Available data are indeed consistent with low collimation, in turn implying a large energy release, comparable to that of long GRBs. The flare properties also constrain the internal shock mechanism, requiring a large Lorentz factor contrast between the colliding shells. This implies that the central engine was active at late times, rather than ejecting all shells simultaneously. The host galaxy has red colors and no ongoing star formation, consistent with previous findings on this GRB. However, it is not a pure elliptical, and has some faint spiral structure. GRB 050724 provides the most compelling case for association between a short burst and a galaxy with old stellar population. It thus plays a pivotal role in constraining progenitors models, which should allow for long delays between birth and explosion."
------------------------------------------------------------
3. [Схожість: 0.6594] UV Star Formation Rates in the Local Universe
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We measure star formation rates of ~50,000 optically-selected galaxies in the local universe (z~0. 1), spanning a range from gas-rich dwarfs to massive ellipticals. We obtain dust-corrected SFRs by fitting the GALEX (UV) and SDSS (optical) photometry to a library of population synthesis models that include dust attenuation. For star-forming galaxies, our UV-based SFRs compare remarkably well with those derived from SDSS H alpha. Deviations from perfect agreement between these two methods are due to differences in the dust attenuation estimates. In contrast to H alpha, UV provides reliable SFRs for galaxies with weak or no H alpha emission, and where H alpha is contaminated with an emission from an AGN. We use full-SED SFRs to calibrate a simple prescription that uses GALEX UV magnitudes to produce good SFRs for normal star-forming galaxies. The specific SFR is considered as a function of stellar mass for (1) star-forming galaxies with no AGN, (2) those hosting an AGN, and for (3) galaxies without H alpha emission. We find that the three have distinct star formation histories, with AGN lying intermediate between the star-forming and the quiescent galaxies. Normal star forming galaxies (without an AGN) lie on a relatively narrow linear sequence. Remarkably, galaxies hosting a strong AGN appear to represent the massive continuation of this sequence. Weak AGN, while also massive, have lower SFR, sometimes extending to the realm of quiescent galaxies. We propose an evolutionary sequence for massive galaxies that smoothly connects normal star-forming galaxies to quiescent (red sequence) galaxies via strong and weak AGN. We confirm that some galaxies with no H alpha emission show signs of SF in the UV. We derive a UV-based cosmic SFR density at z=0. 1 with smaller total error than previous measurements (abridged)."
------------------------------------------------------------
4. [Схожість: 0.6573] A model for the Globular Cluster extreme anomalies
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "In spite of the efforts made in the latest years, still there is no comprehensive explanation for the chemical anomalies of globular cluster stars. Among these, the most striking is oxygen depletion, which reaches values down to [O/Fe]~-0. 4 in most clusters, but in M13 it goes down to less than [O/Fe]~ - 1. In this work we suggest that the anomalies are due to the super position of two different events: 1) PRIMORDIAL SELF-ENRICHMENT: this is asked to explain the oxygen depletion down to a minimum value [O/Fe]~ -0. 4; 2) EXTRA MIXING IN A FRACTION OF THE STARS ALREADY BORN WITH ANOMALOUS COMPOSITION: these objects, starting with already low [O/Fe], will reduce the oxygen abundance down to the most extreme values. Contrary to other models that invoke extra mixing to explain the chemical anomalies, we suggest that it is active only if there is a fraction of the stars in which the primordial composition is not only oxygen depleted, but also extremely helium rich (Y~ 0. 4), as found in a few GCs from their main sequence multiplicity. We propose that the rotational evolution (and an associated extra mixing) of extremely helium rich stars may be affected by the fact that they develop a very small or non existent molecular weight barrier during the evolution. We show that extra mixing in these stars, having initial chemistry that has already been CNO processed, affects mainly the oxygen abundance, and to a much smaller extent if affects the sodium abundance. The model also predicts a large fluorine depletion concomitant with the oxygen depletion, and a further enhancement of the surface helium abundance, which reaches values close to Y=0. 5 in the computed models. We stress that, in this tentative explanation, those stars that are primordially O--depleted, but ARE NOT extremely helium rich do not suffer deep extra mixing."
------------------------------------------------------------
5. [Схожість: 0.6494] Dependence of CMI Growth Rates on Electron Velocity Distributions and
  Perturbation by Solitary Waves
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We calculate growth rates and corresponding gains for RX and LO mode radiation associated with the cyclotron maser instability for parameterized horseshoe electron velocity distributions. The velocity distribution function was modeled to closely fit the electron distribution functions observed in the auroral cavity. We systematically varied the model parameters as well as the propagation direction to study the dependence of growth rates on model parameters. The growth rate depends strongly on loss cone opening angle, which must be less than $90^{o}$ for significant CMI growth. The growth rate is sharply peaked for perpendicular radiation ($k_{\parallel} = 0$), with a full-width at half-maximum $1. 7^{o}$, in good agreement with observed k-vector orientations and numerical simulations. The fractional bandwidth varied between 10$^{-4}$ and 10$^{-2}$, depending most strongly on propagation direction. This range encompasses nearly all observed fractional AKR burst bandwidths. We find excellent agreement between the computed RX mode emergent intensities and observed AKR intensities assuming convective growth length $L_c\approx$20-40 km and group speed 0. 15$c$. The only computed LO mode growth rates compatible observed LO mode radiation levels occurred for number densities more than 100 times the average energetic electron densities measured in auroral cavities. This implies that LO mode radiation is not produced directly by the CMI mechanism but more likely results from mode conversion of RX mode radiation. We find that perturbation of the model velocity distribution by large ion solitary waves (ion holes) can enhance the growth rate by a factor of 2-4. This will result in a gain enhancement more than 40 dB depending on the convective growth length within the structure. Similar enhancements may be caused by EMIC waves."
------------------------------------------------------------
--------------------------------------------------------------------------------
Запит: 'how to mitigate risks in autonomous driving'
--------------------------------------------------------------------------------
--- Результати з індексу: FIXED SIZE CHUNKING ---
1. [Схожість: 0.8099] Improved constraints on dark energy from Chandra X-ray observations of
  the largest relaxed galaxy clusters
Чанк #13.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". (Abridged)"
------------------------------------------------------------
2. [Схожість: 0.8099] Rotation and activity of pre-main-sequence stars
Чанк #12.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". (abridged)"
------------------------------------------------------------
3. [Схожість: 0.8094] The Supernova Channel of Super-AGB Stars
Чанк #12.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "synthetic approach allows us to explore the uncertainty of this number
imposed by uncertainties in the third dredge-up efficiency and ABG mass loss
rate"
------------------------------------------------------------
4. [Схожість: 0.8084] Our Peculiar Motion Away from the Local Void
Чанк #13.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: ". The Local Void is large!"
------------------------------------------------------------
5. [Схожість: 0.8064] Absolute Calibration and Characterization of the Multiband Imaging
  Photometer for Spitzer. II. 70 micron Imaging
Чанк #11.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "validates the MIPS 70 micron operating strategy, especially the use of frequent
stimulator flashes to track the changing responsivities of the Ge:Ga detectors."
------------------------------------------------------------
--- Результати з індексу: SEMANTIC CHUNKING ---
1. [Схожість: 0.7278] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
Чанк #0.0 (Рік: 2007.0, Категорія: cond-mat.stat-mech)
Текст: "In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in spin systems defined on the recurrence lattices. Main tool of our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical mapping (or recursive lattices) for investigation of magnetic properties of the fluid and solid $^3$He, phase transitions in crystals and macromolecules. First, we analyze the helix-coil phase transition for polypeptides and proteins, and describe an quasi unfolding transition (like the cold denaturation process) for the degree of helicity (the order parameter for macromolecules). Next we consider the recurrent models of $^3$He defined on the square, Husimi and hexagon lattices. Using the method of dynamical mapping, the magnetization curves with plateaus, bifurcation point and one period doubling are obtained. Then we investigate the model with cubic symmetry defined on the Bethe lattice and containing both linear and quadratic spin-spin interactions. The magnetization of the system is calculated, and a complex structure of the phase transitions between the disordered, partially ordered and completely ordered states is observed. In the framework of QCD, we consider the azimuthal asymmetries in heavy flavor production in the lepton-nucleon deep inelastic scattering (DIS). We calculate the azimuthal (or $\phi$-) dependence of the next-to-leading order heavy-quark-initiated contributions to DIS. It is shown that, contrary to the basic gluon-initiated component, the photon-quark scattering mechanism is practically $\cos2\phi$-independent. We investigate the possibility of measuring both nonperturbative (intrinsic) and perturbative (CTEQ, MRST) charm distributions using the $\cos2\phi$ asymmetry."
------------------------------------------------------------
2. [Схожість: 0.6887] Multicolor observations of the afterglow of the short/hard GRB 050724
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "New information on short/hard gamma-ray bursts (GRBs) is being gathered thanks to the discovery of their optical and X-ray afterglows. However, some key aspects are still poorly understood, including the collimation level of the outflow, the duration of the central engine activity, and the properties of the progenitor systems. We want to constrain the physical properties of the short GRB 050724 and of its host galaxy, and make some inferences on the global short GRB population. We present optical observations of the afterglow of GRB 050724 and of its host galaxy, significantly expanding the existing dataset for this event. We compare our results with models, complementing them with available measurements from the literature. We study the afterglow light curve and spectrum including X-ray data. We also present observations of the host galaxy. The observed optical emission was likely related to the large flare observed in the X-ray light curve. The apparent steep decay was therefore not due to the jet effect. Available data are indeed consistent with low collimation, in turn implying a large energy release, comparable to that of long GRBs. The flare properties also constrain the internal shock mechanism, requiring a large Lorentz factor contrast between the colliding shells. This implies that the central engine was active at late times, rather than ejecting all shells simultaneously. The host galaxy has red colors and no ongoing star formation, consistent with previous findings on this GRB. However, it is not a pure elliptical, and has some faint spiral structure. GRB 050724 provides the most compelling case for association between a short burst and a galaxy with old stellar population. It thus plays a pivotal role in constraining progenitors models, which should allow for long delays between birth and explosion."
------------------------------------------------------------
3. [Схожість: 0.6843] High energy afterglows and flares from Gamma-Ray Burst by Inverse
  Compton emission
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We perform a detailed study of inverse Compton (IC) emission for a fireball undergoing external shock (ES) in either a uniform or a wind-like interstellar medium, and assess the relative importance of IC and synchrotron emissions. We determine the primary model parameters driving the IC to synchrotron emission ratio in the case of a short duration central engine. We then investigate the case of ES by a long duration central engine, or delayed external shock (DES), a model that can account for some of the flares observed in GRB X-ray light curves. We present model predictions, in particular in terms of GeV vs X-ray behavior, and compare them with other models proposed to explain the origin of flares. We find that if most of the emission occurs when the fireball is in the fast cooling regime, then a substantial GeV emission is expected both for a short (standard ES) and a long (DES) duration central engine activity. In particular, in the context of standard ES we are able to account for the delayed emission observed in GRB940217. In the case of DES, we find that IC scattering of X-ray flare photons can produce high energy flares in the GeV band, which can be detected by GLAST. The detectability of high energy flares improves with the burst kinetic energy: about 30% of Swift GRBs showing flares in their X-ray light curve have sufficiently large kinetic energy so that the expected high flares can be detected by GLAST. One important prediction of the DES model is the simultaneity between low and high energy flares. To test this simultaneity, the peak energies of both flares need to fall below or within the observational bands. We predict that X-ray flares with peak energy of ~10 eV produce high energy flares with peak energy of around 100 MeV-GeV. Observations by Swift and GLAST then, can test the predicted simultaneity."
------------------------------------------------------------
4. [Схожість: 0.6689] Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We here develop an improved way of using a rotating star as a clock, set it using the Sun, and demonstrate that it keeps time well. This technique, called gyrochronology, permits the derivation of ages for solar- and late-type main sequence stars using only their rotation periods and colors. The technique is clarified and developed here, and used to derive ages for illustrative groups of nearby, late-type field stars with measured rotation periods. We first demonstrate the reality of the interface sequence, the unifying feature of the rotational observations of cluster and field stars that makes the technique possible, and extends it beyond the proposal of Skumanich by specifying the mass dependence of rotation for these stars. We delineate which stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their dependence on color and period. Representative age errors associated with the technique are estimated at ~15% (plus possible systematic errors) for late-F, G, K, & early-M stars. Ages derived via gyrochronology for the Mt. Wilson stars are shown to be in good agreement with chromospheric ages for all but the bluest stars, and probably superior. Gyro ages are then calculated for each of the active main sequence field stars studied by Strassmeier and collaborators where other ages are not available. These are shown to be mostly younger than 1Gyr, with a median age of 365Myr. The sample of single, late-type main sequence field stars assembled by Pizzolato and collaborators is then assessed, and shown to have gyro ages ranging from under 100Myr to several Gyr, and a median age of 1. 2Gyr. Finally, we demonstrate that the individual components of the three wide binaries XiBooAB, 61CygAB, & AlphaCenAB yield substantially the same gyro ages."
------------------------------------------------------------
5. [Схожість: 0.6656] A Systematic Study of the Final Masses of Gas Giant Planets
Чанк #0.0 (Рік: 2007.0, Категорія: astro-ph)
Текст: "We construct an analytic model for the rate of gas accretion onto a planet embedded in a protoplanetary disk as a function of planetary mass, disk viscosity, disk scale height, and unperturbed surface density in order to study the long-term accretion and final masses of gas giant planets. We first derive an analytical formula for surface density profile near the planetary orbit from considerations of the balance of force and the dynamical stability. Using it in the empirical formula linking surface density with gas accretion rate that is derived based on hydrodynamic simulations of Tanigawa and Watanabe (2002, ApJ 586, 506), we then simulate the mass evolution of gas giant planets in viscously-evolving disks. We finally determine the final mass as a function of semi-major axis of the planet. We find that the disk can be divided into three regions characterized by different processes by which the final mass is determined. In the inner region, the planet grows quickly and forms a deep gap to suppress the growth by itself before disk dissipation. The final mass of the planet in this region is found to increase with the semi-major axis in a similar way to the mass given by the viscous condition for gap opening, but the former is larger by a factor of approximately 10 than the latter. In the intermediate region, viscous diffusion of the disk gas limits the gas accretion before the planet form a deep gap. The final mass can be up to the disk mass, when disk viscous evolution occurs faster than disk evaporation. In the outer region, planets capture only tiny amounts of gas within the lifetime of the disk to form Neptune-like planets. We derive analytic formulae for the final masses in the different regions and the locations of the boundaries, which are helpful to gain a systematic understanding of the masses of gas giant planets."
------------------------------------------------------------
```
<br>

1. Яка стратегія дає більш осмислені чанки?
Semantic chanking дає більш осмислені чанки

<br>

2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?
Так, у fixed-size chunking часто речення бувають порізані. Це може погано впливати на ембеддинги бо додає шуму у вектор, зміщучи його в багатовимірному просторі в рандомному напрямку від правильної відповіді. 

<br>

3. Як розмір overlap впливає на кількість чанків і покриття тексту?
Чим більший оверлап, тим більшу кількість чанків отримуємо на виході. 
Увесь текст потрапить до бази незалежно від того, який буде оверлап. Але якщо overlap = 0, то на стику чанків речення може бути розірване. 
Якщо overlap дуже великий, то є ймовірність, що деякі речення будуть дублюватись у різних чанках.

<br>

### 🔹 Частина 5 — Гібридний пошук
```
(.venv) PS E:\Projects\goit\NoSQL\hw2> python .\scripts\06_hybrid_search.py
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.


--------------------------------------------------------------------------------
Запит: 'BERT fine-tuning'
--------------------------------------------------------------------------------
--- Топ-5 BM25 (Лексичний пошук) ---
1. [BM25 Score: 11.5017] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess (ID: 0705.4387)
2. [BM25 Score: 9.5047] Fine-Tuning in Brane-antibrane Inflation (ID: 0705.2982)
3. [BM25 Score: 8.0545] Conformal dynamics in gauge theories via non-perturbative
  renormalization group (ID: 0704.3659)
4. [BM25 Score: 7.3391] Inverse Monte-Carlo determination of effective lattice models for SU(3)
  Yang-Mills theory at finite temperature (ID: 0704.2570)
5. [BM25 Score: 6.9883] Eternal Inflation is "Expensive" (ID: 0705.0267)
6. [BM25 Score: 6.8130] String tension and removal of lattice coarsening effects in Monte Carlo
  Renormalization Group (ID: 0705.0383)
7. [BM25 Score: 6.5772] Can the Baryon Number Density and the Cosmological Constant be
  interrelated? (ID: 0705.3682)
8. [BM25 Score: 6.3785] Electroproduction of kaons from the proton in a Regge-plus-resonance
  approach (ID: 0704.3691)
9. [BM25 Score: 6.2321] Natural SUSY Dark Matter: A Window on the GUT Scale (ID: 0706.0031)
10. [BM25 Score: 5.6402] Chasing Brane Inflation in String-Theory (ID: 0705.4682)
--- Топ-5 Pinecone (Векторний пошук) ---
1. [Cosine/Dot: 0.8645] Misere quotients for impartial games: Supplementary material (ID: 0705.2404)
2. [Cosine/Dot: 0.8533] Introduction to Phase Transitions in Random Optimization Problems (ID: 0704.2536)
3. [Cosine/Dot: 0.8500] Abstract Convexity and Cone-Vexing Abstractions (ID: 0705.2793)
4. [Cosine/Dot: 0.8481] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (ID: 0706.0249)
5. [Cosine/Dot: 0.8473] Experimental local realism tests without fair sampling assumption (ID: 0705.0439)
6. [Cosine/Dot: 0.8456] The Call of Mathematics (ID: 0705.4123)
7. [Cosine/Dot: 0.8430] Extracting falsifiable predictions from sloppy models (ID: 0704.3049)
8. [Cosine/Dot: 0.8419] Fluctuation-enhanced sensing (ID: 0705.0160)
9. [Cosine/Dot: 0.8393] A simple algorithm based on fluctuations to play the market (ID: 0705.2097)
10. [Cosine/Dot: 0.8366] Why only few are so successful ? (ID: 0704.2139)
--- Топ-5 HYBRID (RRF: BM25 + Vector) ---
1. [RRF Score: 0.0164] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess (ID: 0705.4387)
2. [RRF Score: 0.0164] Misere quotients for impartial games: Supplementary material (ID: 0705.2404)
3. [RRF Score: 0.0161] Fine-Tuning in Brane-antibrane Inflation (ID: 0705.2982)
4. [RRF Score: 0.0161] Introduction to Phase Transitions in Random Optimization Problems (ID: 0704.2536)
5. [RRF Score: 0.0159] Conformal dynamics in gauge theories via non-perturbative
  renormalization group (ID: 0704.3659)
6. [RRF Score: 0.0159] Abstract Convexity and Cone-Vexing Abstractions (ID: 0705.2793)
7. [RRF Score: 0.0156] Inverse Monte-Carlo determination of effective lattice models for SU(3)
  Yang-Mills theory at finite temperature (ID: 0704.2570)
8. [RRF Score: 0.0156] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (ID: 0706.0249)
9. [RRF Score: 0.0154] Eternal Inflation is "Expensive" (ID: 0705.0267)
10. [RRF Score: 0.0154] Experimental local realism tests without fair sampling assumption (ID: 0705.0439)


--------------------------------------------------------------------------------
Запит: 'Yann LeCun convolutional networks'
--------------------------------------------------------------------------------
--- Топ-5 BM25 (Лексичний пошук) ---
1. [BM25 Score: 13.4827] On Punctured Pragmatic Space-Time Codes in Block Fading Channel (ID: 0704.0282)
2. [BM25 Score: 13.3659] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
  Codes (ID: 0704.1411)
3. [BM25 Score: 8.2349] Response of degree-correlated scale-free networks to stimuli (ID: 0704.1849)
4. [BM25 Score: 7.6366] Numerical evaluation of the upper critical dimension of percolation in
  scale-free networks (ID: 0705.1547)
5. [BM25 Score: 7.5805] On Automorphism Groups of Networks (ID: 0705.3215)
6. [BM25 Score: 7.5789] Optimization in Gradient Networks (ID: 0704.1144)
7. [BM25 Score: 7.5478] Option Pricing Using Bayesian Neural Networks (ID: 0705.1680)
8. [BM25 Score: 7.4290] Analysis of random Boolean networks using the average sensitivity (ID: 0704.0197)
9. [BM25 Score: 7.1858] Anonymity in the Wild: Mixes on unstructured networks (ID: 0706.0430)
10. [BM25 Score: 7.1323] Phase Transitions on Fractals and Networks (ID: 0704.3849)
--- Топ-5 Pinecone (Векторний пошук) ---
1. [Cosine/Dot: 0.8479] Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach (ID: 0705.0211)
2. [Cosine/Dot: 0.8431] The Netsukuku network topology (ID: 0705.0819)
3. [Cosine/Dot: 0.8429] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (ID: 0706.0249)
4. [Cosine/Dot: 0.8346] Modeling the field of laser welding melt pool by RBFNN (ID: 0704.0611)
5. [Cosine/Dot: 0.8314] Adaptive classification of temporal signals in fixed-weights recurrent
  neural networks: an existence proof (ID: 0705.3370)
6. [Cosine/Dot: 0.8285] Optimization in Gradient Networks (ID: 0704.1144)
7. [Cosine/Dot: 0.8276] Misere quotients for impartial games: Supplementary material (ID: 0705.2404)
8. [Cosine/Dot: 0.8272] Multi-Dimensional Recurrent Neural Networks (ID: 0705.2011)
9. [Cosine/Dot: 0.8271] Quantum Shortest Path Netsukuku (ID: 0705.0817)
10. [Cosine/Dot: 0.8270] Flops connect minimal models (ID: 0704.1013)
--- Топ-5 HYBRID (RRF: BM25 + Vector) ---
1. [RRF Score: 0.0303] Optimization in Gradient Networks (ID: 0704.1144)
2. [RRF Score: 0.0164] On Punctured Pragmatic Space-Time Codes in Block Fading Channel (ID: 0704.0282)
3. [RRF Score: 0.0164] Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach (ID: 0705.0211)
4. [RRF Score: 0.0161] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
  Codes (ID: 0704.1411)
5. [RRF Score: 0.0161] The Netsukuku network topology (ID: 0705.0819)
6. [RRF Score: 0.0159] Response of degree-correlated scale-free networks to stimuli (ID: 0704.1849)
7. [RRF Score: 0.0159] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (ID: 0706.0249)
8. [RRF Score: 0.0156] Numerical evaluation of the upper critical dimension of percolation in
  scale-free networks (ID: 0705.1547)
9. [RRF Score: 0.0156] Modeling the field of laser welding melt pool by RBFNN (ID: 0704.0611)
10. [RRF Score: 0.0154] On Automorphism Groups of Networks (ID: 0705.3215)


--------------------------------------------------------------------------------
Запит: 'making computers understand human emotions from text'
--------------------------------------------------------------------------------
--- Топ-5 BM25 (Лексичний пошук) ---
1. [BM25 Score: 18.2706] An Automated Evaluation Metric for Chinese Text Entry (ID: 0704.3662)
2. [BM25 Score: 17.1435] On the Development of Text Input Method - Lessons Learned (ID: 0704.3665)
3. [BM25 Score: 16.6411] Towards Understanding the Origin of Genetic Languages (ID: 0705.3895)
4. [BM25 Score: 12.0869] Detecting anchoring in financial markets (ID: 0705.3319)
5. [BM25 Score: 11.8141] Database Manipulation on Quantum Computers (ID: 0705.4303)
6. [BM25 Score: 11.6820] Philosophy and Relativity (ID: 0705.4441)
7. [BM25 Score: 11.4283] Text Line Segmentation of Historical Documents: a Survey (ID: 0704.1267)
8. [BM25 Score: 11.4243] Using Image Attributes for Human Identification Protocols (ID: 0704.2295)
9. [BM25 Score: 11.3773] Automatically Restructuring Practice Guidelines using the GEM DTD (ID: 0706.1137)
10. [BM25 Score: 11.3135] Lecture notes on Optical Quantum Computing (ID: 0705.4193)
--- Топ-5 Pinecone (Векторний пошук) ---
1. [Cosine/Dot: 0.8287] Opinion Dynamics and Sociophysics (ID: 0705.0891)
2. [Cosine/Dot: 0.8228] On the Development of Text Input Method - Lessons Learned (ID: 0704.3665)
3. [Cosine/Dot: 0.8092] Extracting the hierarchical organization of complex systems (ID: 0705.1679)
4. [Cosine/Dot: 0.8028] Novelty and Collective Attention (ID: 0704.1158)
5. [Cosine/Dot: 0.8021] Narratives within immersive technologies (ID: 0704.2542)
6. [Cosine/Dot: 0.8014] Reaction Time of a Group of Physics Students (ID: 0706.1295)
7. [Cosine/Dot: 0.8012] Introduction to Arabic Speech Recognition Using CMUSphinx System (ID: 0704.2083)
8. [Cosine/Dot: 0.8011] The Answer is Blowing in the Wind (ID: 0705.2228)
9. [Cosine/Dot: 0.8001] Redesigning Computer-based Learning Environments: Evaluation as
  Communication (ID: 0706.1127)
10. [Cosine/Dot: 0.7998] Approximate textual retrieval (ID: 0705.0751)
--- Топ-5 HYBRID (RRF: BM25 + Vector) ---
1. [RRF Score: 0.0323] On the Development of Text Input Method - Lessons Learned (ID: 0704.3665)
2. [RRF Score: 0.0164] An Automated Evaluation Metric for Chinese Text Entry (ID: 0704.3662)
3. [RRF Score: 0.0164] Opinion Dynamics and Sociophysics (ID: 0705.0891)
4. [RRF Score: 0.0159] Towards Understanding the Origin of Genetic Languages (ID: 0705.3895)
5. [RRF Score: 0.0159] Extracting the hierarchical organization of complex systems (ID: 0705.1679)
6. [RRF Score: 0.0156] Detecting anchoring in financial markets (ID: 0705.3319)
7. [RRF Score: 0.0156] Novelty and Collective Attention (ID: 0704.1158)
8. [RRF Score: 0.0154] Database Manipulation on Quantum Computers (ID: 0705.4303)
9. [RRF Score: 0.0154] Narratives within immersive technologies (ID: 0704.2542)
10. [RRF Score: 0.0152] Philosophy and Relativity (ID: 0705.4441)


--------------------------------------------------------------------------------
Запит: 'reinforcement learning'
--------------------------------------------------------------------------------
--- Топ-5 BM25 (Лексичний пошук) ---
1. [BM25 Score: 18.4547] Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa (ID: 0706.0280)
2. [BM25 Score: 10.8613] Ensemble Learning for Free with Evolutionary Algorithms ? (ID: 0704.3905)
3. [BM25 Score: 10.7817] Structure of interacting aggregates of silica nanoparticles in a polymer
  matrix: Small-angle scattering and Reverse Monte-Carlo simulations (ID: 0705.3220)
4. [BM25 Score: 10.6912] An Adaptive Strategy for the Classification of G-Protein Coupled
  Receptors (ID: 0704.3453)
5. [BM25 Score: 10.6711] Statistical Mechanics of Nonlinear On-line Learning for Ensemble
  Teachers (ID: 0705.2318)
6. [BM25 Score: 10.3276] Developing a Collaborative and Autonomous Training and Learning
  Environment for Hybrid Wireless Networks (ID: 0706.1201)
7. [BM25 Score: 10.3144] A mathematical analysis of the effects of Hebbian learning rules on the
  dynamics and structure of discrete-time random recurrent neural networks (ID: 0705.3690)
8. [BM25 Score: 9.5618] A simple spontaneously active Hebbian learning model: homeostasis of
  activity and connectivity, and consequences for learning and epileptogenesis (ID: 0705.3691)
9. [BM25 Score: 9.2833] Validating module network learning algorithms using simulated data (ID: 0705.0666)
10. [BM25 Score: 8.9946] The Parameter-Less Self-Organizing Map algorithm (ID: 0705.0199)
--- Топ-5 Pinecone (Векторний пошук) ---
1. [Cosine/Dot: 0.8445] Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa (ID: 0706.0280)
2. [Cosine/Dot: 0.8194] Introduction to Phase Transitions in Random Optimization Problems (ID: 0704.2536)
3. [Cosine/Dot: 0.8102] Architecture for Pseudo Acausal Evolvable Embedded Systems (ID: 0704.0985)
4. [Cosine/Dot: 0.8010] Why only few are so successful ? (ID: 0704.2139)
5. [Cosine/Dot: 0.7993] Opinion Dynamics and Sociophysics (ID: 0705.0891)
6. [Cosine/Dot: 0.7961] Modeling the field of laser welding melt pool by RBFNN (ID: 0704.0611)
7. [Cosine/Dot: 0.7948] A Study in a Hybrid Centralised-Swarm Agent Community (ID: 0705.2307)
8. [Cosine/Dot: 0.7941] Misere quotients for impartial games: Supplementary material (ID: 0705.2404)
9. [Cosine/Dot: 0.7915] Phase Transitions on Fractals and Networks (ID: 0704.3849)
10. [Cosine/Dot: 0.7910] Quantum Shortest Path Netsukuku (ID: 0705.0817)
--- Топ-5 HYBRID (RRF: BM25 + Vector) ---
1. [RRF Score: 0.0328] Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa (ID: 0706.0280)
2. [RRF Score: 0.0161] Ensemble Learning for Free with Evolutionary Algorithms ? (ID: 0704.3905)
3. [RRF Score: 0.0161] Introduction to Phase Transitions in Random Optimization Problems (ID: 0704.2536)
4. [RRF Score: 0.0159] Structure of interacting aggregates of silica nanoparticles in a polymer
  matrix: Small-angle scattering and Reverse Monte-Carlo simulations (ID: 0705.3220)
5. [RRF Score: 0.0159] Architecture for Pseudo Acausal Evolvable Embedded Systems (ID: 0704.0985)
6. [RRF Score: 0.0156] An Adaptive Strategy for the Classification of G-Protein Coupled
  Receptors (ID: 0704.3453)
7. [RRF Score: 0.0156] Why only few are so successful ? (ID: 0704.2139)
8. [RRF Score: 0.0154] Statistical Mechanics of Nonlinear On-line Learning for Ensemble
  Teachers (ID: 0705.2318)
9. [RRF Score: 0.0154] Opinion Dynamics and Sociophysics (ID: 0705.0891)
10. [RRF Score: 0.0152] Developing a Collaborative and Autonomous Training and Learning
  Environment for Hybrid Wireless Networks (ID: 0706.1201)
```

<br>

1. Який метод дав кращий результат і чому?
Гібрідний пошук, мабуть, є кращим. Він враховує і зміст, і лексичну складову і видає більш зважений результат.

<br>

2. Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?
Так, є. BM25 шукає в тексті вказані в запиті слова, векторний пошук буде більш прив'язан до зміста, навіть якщо слова в запиті в тексті не зустрічались. Гібрідний пошук видає більш збалансований результат - туди можуть потрапити статті з кожного з попередніх списків, деякі статті можуть не потрапити взагалі.  

<br>

3. Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?
Параметр k визначає як швидко змінюється вага документа при зниженні його позиції у видачі. 
Якщо k=1 - вага документа знижується дуже швидко. В такому випадку гібридний алгоритм буде відавати перевагу тим документам, що зайняли 1 місце хоча б в одному з алгоритмів (BM25, вектор).
Якщо k=60 - вага документа знижується повільно. В такому випадку гібридний алгоритм буде віддавати перевагу тим документам, які обидва алгоритма вважать релевантним.

<br>

### 🔹 Частина 6 — Аналіз і висновки
1. Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?
BM25 - лексичний пошук, може бути найбільш корисним, коли щось шукаєш по специфічним ключовим словам(ім'я автора, спеціфіний термін тощо).
Векторний/семантичний пошук може бути більш корисним, коли запит написан природньою мовою, є загальні описові запити, запити іншою мовою. Тоді результат пошуку буде залежити зміста і контекста, а не від наявності конкретних слів в запиті.
В роботі по запиту "reinforcement learning" обидва алгоритма видали 1 місщем одну і ту саму статтю: Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa(ID: 0706.0280)

<br>

2. Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?
Якщо дуже маленький, це може привести до того, текст буде розділений на окремі фрази/речення, а зміст або контекст буде втрачатися. Також деякі речення можуть не влізти в чанк і тоді вони будуть обрубатися. 
Якщо чанк занадто великий, то він може охоплювати декілька підтем/думок. Через це вектор буде складніше асоціювати із якоюсь конкретною темою і він може загубитись у багатовимірному просторі. Такий великий чанк буде гірше з'являтись в результатам пошуку. 

Маленький чанк (10-15 слів), мабуть, краще використовувати, якщо пошук відбувається в якомусь словнику при пошуку точних фактів, там де інформація викладається дуже стисло. 
Великі чанки доцільно використовувати при пошуку у великих текстах(наукові статті, художня література тощо), тобто там, де одна думка може буде розтікатись на 3 сторінки.
А якщо точно не зрозуміло який чанк обрати, то можна взяти чанк по дефолту ~100-300 слів.

<br>

3. Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.

Косінусна схожість для нормалізованих векторів дорінює скалярному добутку (як це визначили в п1.3.) 
$Cosine(A, B) = Dot Product(A, B) = A \cdot B$

$L2^2 = ||Q - D||^2 = (Q - D) \cdot (Q - D) = (Q - D) \cdot (Q - D) = (Q \cdot Q) - 2(Q \cdot D) + (D \cdot D) = 1 - 2(Q \cdot D) + 1 = 2 - 2(Q \cdot D) = 2 - 2 \cdot Cosine(Q, D)$

Це означає, що на нормалізованих векторах обидві метрики дадуть однаковий порядок ранжування результатів — вектор із мінімальною L2-відстанню також матиме максимальну косинусну схожість.

Тобто, якщо б ми створили індекс з метрикою euclidean (L2), нічого страшного не трапилось би і пошук видавав би такий самий результат.

<br>

4. Обмеження Pinecone Starter. З якими обмеженнями  безкоштовного  тіру  ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?

Я не стикалася з обмеженнями, але теоретично можна спіймати такі проблеми як:
- якщо завантажувати дані на Pinecone великими батчами в декілька потоків, то Pinecone може видавати помилку Too Many Requests.
- обмеження по об'єму пам'яті в 2Гб. Якщо вийти за цей ліміт, також може бути помилка.


Зараз текст зберігається в метаданих Pinecone. Якби датасет був на 10 млн статтей, то можливо було б доцільно відокремити текстові данні і помістити в якусь іншу базу(SQL або NoSQL). 
Також треба було б подумати або про роширення тарифного плану на Pinecone, або перехід на власний сервер на Qdrant, бо ресурсів бескоштовного тіру, мабуть, вже не буде вистачати.  

<br>