# NetML Weekly — Week of February 05, 2026

*67 papers at the intersection of machine learning and computer networking.*

## Sessions

- [Information Theory](#information-theory) (3 papers)
- [Networking](#networking) (12 papers)
- [eess.SY](#eess.sy) (2 papers)
- [Security & Privacy](#security-privacy) (10 papers)
- [Machine Learning](#machine-learning) (21 papers)
- [cs.HC](#cs.hc) (1 papers)
- [stat.ME](#stat.me) (1 papers)
- [cs.MM](#cs.mm) (1 papers)
- [cs.RO](#cs.ro) (1 papers)
- [Signal Processing](#signal-processing) (6 papers)
- [astro-ph.IM](#astro-ph.im) (1 papers)
- [math-ph](#math-ph) (1 papers)
- [physics.geo-ph](#physics.geo-ph) (1 papers)
- [quant-ph](#quant-ph) (1 papers)
- [cs.CV](#cs.cv) (3 papers)
- [eess.IV](#eess.iv) (1 papers)
- [physics.chem-ph](#physics.chem-ph) (1 papers)

---

## Information Theory

*Papers in Information Theory*

### Predictive-State Communication: Innovation Coding and Reconciliation under Delay

**Authors:** Ozgur Ercetin, Mohaned Chraiti
[arXiv](http://arxiv.org/abs/2602.10542v1) · [PDF](https://arxiv.org/pdf/2602.10542v1)

Shannon theory models communication as the reliable transfer of symbol sequences, with performance governed by capacity and rate-distortion limits. When both endpoints possess strong predictors -- as in modern large language models and related generative priors -- literal symbol transport is no longer the only operational regime. We propose predictive-state communication (PSC), in which the transmitter and receiver maintain an explicit shared predictive state, and the physical channel is used primarily to convey innovations, i.e., corrective information that reconciles the receiver's provisional trajectory with the transmitter's realized trajectory. This viewpoint replaces entropy-rate accounting by cross-entropy accounting under model mismatch, and it introduces feasibility constraints that depend jointly on capacity, delay, and perceptual continuity requirements; the resulting operating set is typically a bounded perception-capacity band rather than a one-sided threshold. We outline the protocol and architectural implications (state identifiers, anchors, bounded rollback, and patch-based updates) and provide a stylized illustrative example to visualize the induced feasibility region and its dependence on predictive quality.

### Multipoint Code-Weight Sphere Decoding: Parallel Near-ML Decoding for Short-Blocklength Codes

**Authors:** Yubeen Jo, Geon Choi, Yongjune Kim, Namyoon Lee
[arXiv](http://arxiv.org/abs/2602.08501v1) · [PDF](https://arxiv.org/pdf/2602.08501v1)

Ultra-reliable low-latency communications (URLLC) operate with short packets, where finite-blocklength effects make near-maximum-likelihood (near-ML) decoding desirable but often too costly. This paper proposes a two-stage near-ML decoding framework that applies to any linear block code. In the first stage, we run a low-complexity decoder to produce a candidate codeword and a cyclic redundancy check. When this stage succeeds, we terminate immediately. When it fails, we invoke a second-stage decoder, termed multipoint code-weight sphere decoding (MP-WSD). The central idea behind {MP-WSD} is to concentrate the ML search where it matters. We pre-compute a set of low-weight codewords and use them to generate structured local perturbations of the current estimate. Starting from the first-stage output, MP-WSD iteratively explores a small Euclidean sphere of candidate codewords formed by adding selected low-weight codewords, tightening the search region as better candidates are found. This design keeps the average complexity low: at high signal-to-noise ratio, the first stage succeeds with high probability and the second stage is rarely activated; when it is activated, the search remains localized. Simulation results show that the proposed decoder attains near-ML performance for short-blocklength, low-rate codes while maintaining low decoding latency.

### Deep Variable-Length Feedback Codes

**Authors:** Yu Ding, Yulin Shao
[arXiv](http://arxiv.org/abs/2602.07881v1) · [PDF](https://arxiv.org/pdf/2602.07881v1)

Deep learning has enabled significant advances in feedback-based channel coding, yet existing learned schemes remain fundamentally limited: they employ fixed block lengths, suffer degraded performance at high rates, and cannot fully exploit the adaptive potential of feedback. This paper introduces Deep Variable-Length Feedback (DeepVLF) coding, a flexible coding framework that dynamically adjusts transmission length via learned feedback. We propose two complementary architectures: DeepVLF-R, where termination is receiver-driven, and DeepVLF-T, where the transmitter controls termination. Both architectures leverage bit-group partitioning and transformer-based encoder-decoder networks to enable fine-grained rate adaptation in response to feedback. Evaluations over AWGN and 5G-NR fading channels demonstrate that DeepVLF substantially outperforms state-of-the-art learned feedback codes. It achieves the same block error rate with 20%-55% fewer channel uses and lowers error floors by orders of magnitude, particularly in high-rate regimes. Encoding dynamics analysis further reveals that the models autonomously learn a two-phase strategy analogous to classical Schalkwijk-Kailath coding: an initial information-carrying phase followed by a noise-cancellation refinement phase. This emergent behavior underscores the interpretability and information-theoretic alignment of the learned codes.

---

## Networking

*Papers in Networking*

### To Reconfigure or Not to Reconfigure: Optimizing All-to-All Collectives in Circuit-Switched Photonic Interconnects

**Authors:** Anchengcheng Zhou, Vamsi Addanki, Maria Apostolaki
[arXiv](http://arxiv.org/abs/2602.10468v1) · [PDF](https://arxiv.org/pdf/2602.10468v1)

All-to-all collective communication is a core primitive in distributed machine learning and high-performance computing. At the server scale, the communication demands of these workloads are increasingly outstripping the bandwidth and energy limits of electrical interconnects, driving a growing interest in photonic interconnects. However, leveraging these interconnects for all-to-all communication is nontrivial. The core challenge lies in jointly optimizing a sequence of topologies and flow schedules, reconfiguring only when the transmission savings from traversing shorter paths outweigh the reconfiguration cost. Yet the search space of this joint optimization is enormous. Existing work sidesteps this challenge by making unrealistic assumptions on reconfiguration costs so that it is never or always worthwhile to reconfigure. In this paper, we show that any candidate sequence of topologies and flow schedules can be expressed as a sum of adjacency matrices and their powers. This abstraction captures the entire solution space and yields a lower bound on all-to-all completion time. Building on this formulation, we identify a family of topology sequences with strong symmetry and high expansion that admits bandwidth-efficient schedules, which our algorithm constructs with low computational overhead. Together, these insights allow us to efficiently construct near-optimal solutions, effectively avoiding enumeration of the combinatorial design space. Evaluation shows that our approach reduces all-to-all completion time by up to 44% on average across a wide range of network parameters, message sizes and workload types.

### Scaling Routers with In-Package Optics and High-Bandwidth Memories

**Authors:** Isaac Keslassy, Ilay Yavlovich, Jose Yallouz, Tzu-Chien Hsueh, Yeshaiahu Fainman, Bill Lin
[arXiv](http://arxiv.org/abs/2602.10505v1) · [PDF](https://arxiv.org/pdf/2602.10505v1)

This paper aims to apply two major scaling transformations from the computing packaging industry to internet routers: the heterogeneous integration of high-bandwidth memories (HBMs) and chiplets, as well as in-package optics. We propose a novel internet router architecture that employs these technologies to achieve a petabit/sec router within a single integrated package. At the top-level, we introduce a novel split-parallel switch architecture that spatially divides (without processing) the incoming fibers and distributes them across smaller independent switches without intermediate OEO conversions or fine-tuned per-packet load-balancing. This passive spatial division enables scaling at the cost of a coarser traffic load balancing. Yet, through extensive evaluations of backbone network traffic, we demonstrate that differences with fine-tuned approaches are small. In addition, we propose a novel HBM-based shared-memory architecture for the implementation of the smaller independent switches, and we introduce a novel parallel frame interleaving algorithm that packs traffic into frames so that HBM banks are accessed at peak HBM data rates in a cyclical interleaving manner. We further discuss why these new technologies represent a paradigm shift in the design of future internet routers. Finally, we emphasize that power consumption may constitute the primary bottleneck to scaling.

### Supercharging Packet-level Network Simulation of Large Model Training via Memoization and Fast-Forwarding

**Authors:** Fei Long, Kaihui Gao, Li Chen, Dan Li, Yiwei Zhang, Fei Gui, Yitao Xing, Wenjia Wei, Bingyang Liu
[arXiv](http://arxiv.org/abs/2602.10615v1) · [PDF](https://arxiv.org/pdf/2602.10615v1)

Packet-level discrete-event simulation (PLDES) is a prevalent tool for evaluating detailed performance of large model training. Although PLDES offers high fidelity and generality, its slow performance has plagued networking practitioners. Existing optimization techniques either simplify the network model, resulting in large errors; or execute it in parallel using multiple processors, with an upper bound on speedup. This paper explores an alternative optimization direction that reduces the computational loads of PLDES while maintaining high fidelity. Our key insight is that, in distributed LLM training, packet-level traffic behaviors often exhibit repetitive contention patterns and steady-states where flow rates stabilize, ignoring these redundant discrete events speeds up the simulation considerably and the error is negligible. We realize this idea by proposing Wormhole, a user-transparent PLDES kernel capable of automatically memoization for unsteady-states and skipping for steady-states. Wormhole adopts network partitioning, state memoization and reuse, and rate-based steady-state identification to accurately determine the periods of each flow's steady-state, while maintaining simulation consistency after fast-forwarding. Experiments demonstrate that Wormhole can achieve a 744x speedup over the original ns-3 (510x for MoE workload), with a bounded error of <1%. Applying current multithreading parallel techniques and Wormhole together allows a 1012x speedup, reducing the simulation time for one GPT-13B training under 128 GPUs from 9 hours to 5 minutes.

### Hybrid Responsible AI-Stochastic Approach for SLA Compliance in Multivendor 6G Networks

**Authors:** Emanuel Figetakis, Ahmed Refaey Hussein
[arXiv](http://arxiv.org/abs/2602.09841v1) · [PDF](https://arxiv.org/pdf/2602.09841v1)

The convergence of AI and 6G network automation introduces new challenges in maintaining transparency, fairness, and accountability across multivendor management systems. Although closed-loop AI orchestration improves adaptability and self-optimization, it also creates a responsibility gap, where violations of SLAs cannot be causally attributed to specific agents or vendors. This paper presents a hybrid responsible AI-stochastic learning framework that embeds fairness, robustness, and auditability directly into the network control loop. The framework integrates RAI games with stochastic optimization, enabling dynamic adversarial reweighting and probabilistic exploration across heterogeneous vendor domains. An RAAP continuously records AI-driven decision trajectories and produces dual accountability reports: user-level SLA summaries and operator-level responsibility analytics. Experimental evaluations on synthetic two-class multigroup datasets demonstrate that the proposed hybrid model improves the accuracy of the worst group by up to 10.5\%. Specifically, hybrid RAI achieved a WGAcc of 60.5\% and an AvgAcc of 72.7\%, outperforming traditional RAI-GA (50.0\%) and ERM (21.5\%). The audit mechanism successfully traced 99\% simulated SLA violations to the AI entities responsible, producing both vendor and agent-level accountability indices. These results confirm that the proposed hybrid approach enhances fairness and robustness as well as establishes a concrete accountability framework for autonomous SLA assurance in multivendor 6G networks.

### PACC: Protocol-Aware Cross-Layer Compression for Compact Network Traffic Representation

**Authors:** Zhaochen Guo, Tianyufei Zhou, Honghao Wang, Ronghua Li, Shinan Liu
[arXiv](http://arxiv.org/abs/2602.08331v1) · [PDF](https://arxiv.org/pdf/2602.08331v1)

Network traffic classification is a core primitive for network security and management, yet it is increasingly challenged by pervasive encryption and evolving protocols. A central bottleneck is representation: hand-crafted flow statistics are efficient but often too lossy, raw-bit encodings can be accurate but are costly, and recent pre-trained embeddings provide transfer but frequently flatten the protocol stack and entangle signals across layers. We observe that real traffic contains substantial redundancy both across network layers and within each layer; existing paradigms do not explicitly identify and remove this redundancy, leading to wasted capacity, shortcut learning, and degraded generalization. To address this, we propose PACC, a redundancy-aware, layer-aware representation framework. PACC treats the protocol stack as multi-view inputs and learns compact layer-wise projections that remain faithful to each layer while explicitly factorizing representations into shared (cross-layer) and private (layer-specific) components. We operationalize these goals with a joint objective that preserves layer-specific information via reconstruction, captures shared structure via contrastive mutual-information learning, and maximizes task-relevant information via supervised losses, yielding compact latents suitable for efficient inference. Across datasets covering encrypted application classification, IoT device identification, and intrusion detection, PACC consistently outperforms feature-engineered and raw-bit baselines. On encrypted subsets, it achieves up to a 12.9% accuracy improvement over nPrint. PACC matches or surpasses strong foundation-model baselines. At the same time, it improves end-to-end efficiency by up to 3.16x.

### NeuroScaler: Towards Energy-Optimal Autoscaling for Container-Based Services

**Authors:** Alisson O. Chaves, Rodrigo Moreira, Larissa F. Rodrigues Moreira, Joao Correia, David Santos, Rui Silva, Tiago Barros, Daniel Corujo, Miguel Rocha, Flavio de Oliveira Silva
[arXiv](http://arxiv.org/abs/2602.08191v1) · [PDF](https://arxiv.org/pdf/2602.08191v1)

Future networks must meet stringent requirements while operating within tight energy and carbon constraints. Current autoscaling mechanisms remain workload-centric and infrastructure-siloed, and are largely unaware of their environmental impact. We present NeuroScaler, an AI-native, energy-efficient, and carbon-aware orchestrator for green cloud and edge networks. NeuroScaler aggregates multi-tier telemetry, from Power Distribution Units (PDUs) through bare-metal servers to virtualized infrastructure with containers managed by Kubernetes, using distinct energy and computing metrics at each tier. It supports several machine learning pipelines that link load, performance, and power. Within this unified observability layer, a model-predictive control policy optimizes energy use while meeting service-level objectives. In a real testbed with production-grade servers supporting real services, NeuroScaler reduces energy consumption by 34.68% compared to the Horizontal Pod Autoscaler (HPA) while maintaining target latency.

### MonkeyTree: Near-Minimal Congestion for Multi-tenant Training via Migration

**Authors:** Anton A. Zabreyko, Weiyang Wang, Manya Ghobadi
[arXiv](http://arxiv.org/abs/2602.08296v2) · [PDF](https://arxiv.org/pdf/2602.08296v2)

We present MonkeyTree, the first system to mitigate network congestion in multi-tenant GPU clusters through job-migration based defragmentation rather than network-layer techniques. As cloud operators co-locate ML training jobs on shared, oversubscribed networks, congestion degrades training throughput for over a third of jobs. Prior approaches either rely on routing and flow scheduling--which we show have fundamental limits when traffic exceeds capacity--or require costly full-bisection bandwidth topologies with packet spraying.
  MonkeyTree exploits characteristics of ML training traffic: ring-based collectives generate exactly one cross-rack flow per rack a job spans, making congestion-free placements achievable. The sparse constraint structure admits abundant valid configurations, making them easy to reach with few migrations. Once reached, low fragmentation is self-reinforcing, as new arrivals disturb only a few racks. MonkeyTree formulates defragmentation as an integer linear program that minimizes worker movements, subject to per-rack fragmentation bounds. We prove a tight bound showing any placement can be defragmented to at most two cross-rack fragments per ToR, and extend the formulation to hybrid parallelism with multiple rings per server. Migration is implemented via in-memory checkpoint-and-restore over RDMA, incurring only 9.02 seconds of system overhead end-to-end per worker. We evaluate MonkeyTree using a custom simulator modeling clusters of up to 2,048 H200 GPUs and prototype on a five-node A100 testbed. MonkeyTree improves average job completion time by 14 percent over the next best baseline on a cluster of 1,024 GPUs with a 4:1 oversubscription. With a high 16:1 oversubscription ratio and 2,048 GPUs, MonkeyTree keeps p99 job completion time within 5 percent of ideal.

### Trajectory-Aware Multi-RIS Activation and Configuration: A Riemannian Diffusion Method

**Authors:** Kaining Wang, Bo Yang, Yusheng Lei, Zhibo Li, Zhiwen Yu, Xuelin Cao, Bin Guo, George C. Alexandropoulos, Dusit Niyato, Mérouane Debbah, Zhu Han
[arXiv](http://arxiv.org/abs/2602.07937v1) · [PDF](https://arxiv.org/pdf/2602.07937v1)

Reconfigurable intelligent surfaces (RISs) offer a low-cost, energy-efficient means for enhancing wireless coverage. Yet, their inherently programmable reflections may unintentionally amplify interference, particularly in large-scale, multi-RIS-enabled mobile communication scenarios where dense user mobility and frequent line-of-sight overlaps can severely degrade the signal-to-interference-plus-noise ratio (SINR). To address this challenge, this paper presents a novel generative multi-RIS control framework that jointly optimizes the ON/OFF activation patterns of multiple RISs in the smart wireless environment and the phase configurations of the activated RISs based on predictions of multi-user trajectories and interference patterns. We specially design a long short-term memory (LSTM) artificial neural network, enriched with speed and heading features, to forecast multi-user trajectories, thereby enabling reconstruction of future channel state information. To overcome the highly nonconvex nature of the multi-RIS control problem, we develop a Riemannian diffusion model on the torus to generate geometry-consistent phase-configuration, where the reverse diffusion process is dynamically guided by reinforcement learning. We then rigorously derive the optimal ON/OFF states of the metasurfaces by comparing predicted achievable rates under RIS activation and deactivation conditions. Extensive simulations demonstrate that the proposed framework achieves up to 30\% SINR improvement over learning-based control and up to 44\% gain compared with the RIS always-on scheme, while consistently outperforming state-of-the-art baselines across different transmit powers, RIS configurations, and interference densities.

### Makespan Minimization in Split Learning: From Theory to Practice

**Authors:** Robert Ganian, Fionn Mc Inerney, Dimitra Tsigkari
[arXiv](http://arxiv.org/abs/2602.06693v1) · [PDF](https://arxiv.org/pdf/2602.06693v1)

Split learning recently emerged as a solution for distributed machine learning with heterogeneous IoT devices, where clients can offload part of their training to computationally-powerful helpers. The core challenge in split learning is to minimize the training time by jointly devising the client-helper assignment and the schedule of tasks at the helpers. We first study the model where each helper has a memory cardinality constraint on how many clients it may be assigned, which represents the case of homogeneous tasks. Through complexity theory, we rule out exact polynomial-time algorithms and approximation schemes even for highly restricted instances of this problem. We complement these negative results with a non-trivial polynomial-time 5-approximation algorithm. Building on this, we then focus on the more general heterogeneous task setting considered by Tirana et al. [INFOCOM 2024], where helpers have memory capacity constraints and clients have variable memory costs. In this case, we prove that, unless P=NP, the problem cannot admit a polynomial-time approximation algorithm for any approximation factor. However, by adapting our aforementioned 5-approximation algorithm, we develop a novel heuristic for the heterogeneous task setting and show that it outperforms heuristics from prior works through extensive experiments.

### Talk Like a Packet: Rethinking Network Traffic Analysis with Transformer Foundation Models

**Authors:** Samara Mayhoub, Chuan Heng Foh, Mahdi Boloursaz Mashhadi, Mohammad Shojafar, Rahim Tafazolli
[arXiv](http://arxiv.org/abs/2602.06636v1) · [PDF](https://arxiv.org/pdf/2602.06636v1)

Inspired by the success of Transformer-based models in natural language processing, this paper investigates their potential as foundation models for network traffic analysis. We propose a unified pre-training and fine-tuning pipeline for traffic foundation models. Through fine-tuning, we demonstrate the generalizability of the traffic foundation models in various downstream tasks, including traffic classification, traffic characteristic prediction, and traffic generation. We also compare against non-foundation baselines, demonstrating that the foundation-model backbones achieve improved performance. Moreover, we categorize existing models based on their architecture, input modality, and pre-training strategy. Our findings show that these models can effectively learn traffic representations and perform well with limited labeled datasets, highlighting their potential in future intelligent network analysis systems.

### Performance Evaluation of V2X Communication Using Large-Scale Traffic Data

**Authors:** John Pravin Arockiasamy, Alexey Vinel
[arXiv](http://arxiv.org/abs/2602.07244v1) · [PDF](https://arxiv.org/pdf/2602.07244v1)

Vehicular communication (V2X) technologies are widely regarded as a cornerstone for cooperative and automated driving, yet their large-scale real-world deployment remains limited. As a result, understanding V2X performance under realistic, full-scale traffic conditions continues to be relevant. Most existing performance evaluations rely on synthetic traffic scenarios generated by simulators, which, while useful, may not fully capture the features of real-world traffic. In this paper, we present a large-scale, data-driven evaluation of V2X communication performance using real-world traffic datasets. Vehicle trajectories derived from the Highway Drone (HighD) and Intersection Drone (InD) datasets are converted into simulation-ready formats and coupled with a standardized V2X networking stack to enable message-level performance analysis for entire traffic populations comprising over hundred thousands vehicles across multiple locations. We evaluate key V2X performance indicators, including inter-generation gap, inter-packet gap, packet delivery ratio, and channel busy ratio, across both highway and urban intersection environments. Our results show that cooperative awareness services remain feasible at scale under realistic traffic conditions. In addition, the findings highlight how traffic density, mobility patterns, and communication range influence V2X performance and how synthetic traffic assumptions may overestimate channel congestion.

### Data analysis of cloud virtualization experiments

**Authors:** Pedro R. X. do Carmo, Eduardo Freitas, Assis T. de Oliveira Filho, Judith Kelner, Djamel Sadok
[arXiv](http://arxiv.org/abs/2602.05792v1) · [PDF](https://arxiv.org/pdf/2602.05792v1)

The cloud computing paradigm underlines data center and telecommunication infrastructure design. Heavily leveraging virtualization, it slices hardware and software resources into smaller software units for greater flexibility of manipulation. Given the considerable benefits, several virtualization forms, with varying processing and communication overheads, emerged, including Full Virtualization and OS Virtualization. As a result, predicting packet throughput at the data plane turns out to be more challenging due to the additional virtualization overhead located at CPU, I/O, and network resources. This research presents a dataset of active network measurements data collected while varying various network parameters, including CPU affinity, frequency of echo packet injection, type of virtual network driver, use of CPU, I/O, or network load, and the number of concurrent VMs. The virtualization technologies used in the study include KVM, LXC, and Docker. The work examines their impact on a key network metric, namely, end-to-end latency. Also, it builds data models to evaluate the impact of a cloud computing environment on packet round-trip time. To explore data visualization, the dataset was submitted to pre-processing, correlation analysis, dimensionality reduction, and clustering. In addition, this paper provides a brief analysis of the dataset, demonstrating its use in developing machine learning-based systems for administrator decision-making.

---

## eess.SY

*Papers in eess.SY*

### Anomaly Detection with Machine Learning Algorithms in Large-Scale Power Grids

**Authors:** Marc Gillioz, Guillaume Dubuis, Étienne Voutaz, Philippe Jacquod
[arXiv](http://arxiv.org/abs/2602.10888v1) · [PDF](https://arxiv.org/pdf/2602.10888v1)

We apply several machine learning algorithms to the problem of anomaly detection in operational data for large-scale, high-voltage electric power grids. We observe important differences in the performance of the algorithms. Neural networks typically outperform classical algorithms such as k-nearest neighbors and support vector machines, which we explain by the strong contextual nature of the anomalies. We show that unsupervised learning algorithm work remarkably well and that their predictions are robust against simultaneous, concurring anomalies.

### EExApp: GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN

**Authors:** Jie Lu, Peihao Yan, Huacheng Zeng
[arXiv](http://arxiv.org/abs/2602.09206v1) · [PDF](https://arxiv.org/pdf/2602.09206v1)

With over 3.5 million 5G base stations deployed globally, their collective energy consumption (projected to exceed 131 TWh annually) raises significant concerns over both operational costs and environmental impacts. In this paper, we present EExAPP, a deep reinforcement learning (DRL)-based xApp for 5G Open Radio Access Network (O-RAN) that jointly optimizes radio unit (RU) sleep scheduling and distributed unit (DU) resource slicing. EExAPP uses a dual-actor-dual-critic Proximal Policy Optimization (PPO) architecture, with dedicated actor-critic pairs targeting energy efficiency and quality-of-service (QoS) compliance. A transformer-based encoder enables scalable handling of variable user equipment (UE) populations by encoding all-UE observations into fixed-dimensional representations. To coordinate the two optimization objectives, a bipartite Graph Attention Network (GAT) is used to modulate actor updates based on both critic outputs, enabling adaptive tradeoffs between power savings and QoS. We have implemented EExAPP and deployed it on a real-world 5G O-RAN testbed with live traffic, commercial RU and smartphones. Extensive over-the-air experiments and ablation studies confirm that EExAPP significantly outperforms existing methods in reducing the energy consumption of RU while maintaining QoS.

---

## Security & Privacy

*Papers in Security & Privacy*

### SecureScan: An AI-Driven Multi-Layer Framework for Malware and Phishing Detection Using Logistic Regression and Threat Intelligence Integration

**Authors:** Rumman Firdos, Aman Dangi
[arXiv](http://arxiv.org/abs/2602.10750v1) · [PDF](https://arxiv.org/pdf/2602.10750v1)

The growing sophistication of modern malware and phishing campaigns has diminished the effectiveness of traditional signature-based intrusion detection systems. This work presents SecureScan, an AI-driven, triple-layer detection framework that integrates logistic regression-based classification, heuristic analysis, and external threat intelligence via the VirusTotal API for comprehensive triage of URLs, file hashes, and binaries. The proposed architecture prioritizes efficiency by filtering known threats through heuristics, classifying uncertain samples using machine learning, and validating borderline cases with third-party intelligence. On benchmark datasets, SecureScan achieves 93.1 percent accuracy with balanced precision (0.87) and recall (0.92), demonstrating strong generalization and reduced overfitting through threshold-based decision calibration. A calibrated threshold and gray-zone logic (0.45-0.55) were introduced to minimize false positives and enhance real-world stability. Experimental results indicate that a lightweight statistical model, when augmented with calibrated verification and external intelligence, can achieve reliability and performance comparable to more complex deep learning systems.

### Beyond Permissions: An Empirical Static Analysis of Privacy and Security Risks in Children-Oriented and General-Audience Mobile Apps for Gaming

**Authors:** Bakheet Aljedaani
[arXiv](http://arxiv.org/abs/2602.10877v1) · [PDF](https://arxiv.org/pdf/2602.10877v1)

Mobile gaming applications (apps) have become increasingly pervasive, including a growing number of games designed for children. Despite their popularity, these apps often integrate complex analytics, advertising, and attribution infrastructures that may introduce privacy and security risks. Existing research has primarily focused on tracking behaviors or monetization models, leaving configuration-level privacy exposure and children-oriented apps underexplored. In this study, we conducted a comparative static analysis of Android mobile games to investigate privacy and security risks beyond permission usage. The analysis follows a three-phase methodology comprising (i) designing study protocol, (ii) Android Package Kit (APK) collection and static inspection, and (iii) data analysis. We examined permissions, manifest-level configuration properties (e.g., backup settings, cleartext network traffic, and exported components), and embedded third-party Software Development Kit (SDK) ecosystems across children-oriented and general-audience mobile games. The extracted indicators are synthesized into qualitative privacy-risk categories to support comparative reporting. The results showed that while children-oriented games often request fewer permissions, they frequently exhibit configuration-level risks and embed third-party tracking SDKs similar to general-audience games. Architectural and configuration decisions play a critical role in shaping privacy risks, particularly for apps targeting children. This study contributes a holistic static assessment of privacy exposure in mobile games and provides actionable insights for developers, platform providers, and researchers seeking to improve privacy-by-design practices in mobile applications.

### Evasion of IoT Malware Detection via Dummy Code Injection

**Authors:** Sahar Zargarzadeh, Mohammad Islam
[arXiv](http://arxiv.org/abs/2602.08170v1) · [PDF](https://arxiv.org/pdf/2602.08170v1)

The Internet of Things (IoT) has revolutionized connectivity by linking billions of devices worldwide. However, this rapid expansion has also introduced severe security vulnerabilities, making IoT devices attractive targets for malware such as the Mirai botnet. Power side-channel analysis has recently emerged as a promising technique for detecting malware activity based on device power consumption patterns. However, the resilience of such detection systems under adversarial manipulation remains underexplored.
  This work presents a novel adversarial strategy against power side-channel-based malware detection. By injecting structured dummy code into the scanning phase of the Mirai botnet, we dynamically perturb power signatures to evade AI/ML-based anomaly detection without disrupting core functionality. Our approach systematically analyzes the trade-offs between stealthiness, execution overhead, and evasion effectiveness across multiple state-of-the-art models for side-channel analysis, using a custom dataset collected from smartphones of diverse manufacturers. Experimental results show that our adversarial modifications achieve an average attack success rate of 75.2\%, revealing practical vulnerabilities in power-based intrusion detection frameworks.

### ICBAC: an Intelligent Contract-Based Access Control framework for supply chain management by integrating blockchain and federated learning

**Authors:** Sadegh Sohani, Salar Ghazi, Farnaz Kamranfar, Sahar Pilehvar Moakhar, Mohammad Allahbakhsh, Haleh Amintoosi, Kaiwen Zhang
[arXiv](http://arxiv.org/abs/2602.08014v1) · [PDF](https://arxiv.org/pdf/2602.08014v1)

This paper addresses the critical challenge of access control in modern supply chains, which operate across multiple independent and competing organizations. Existing access control is static and centralized, unable to adapt to insider threats or evolving contexts. Blockchain improves decentralization but lacks behavioral intelligence, while centralized machine learning for anomaly detection requires aggregating sensitive data, violating privacy.
  The proposed solution is ICBAC, an intelligent contract-based access control framework. It integrates permissioned blockchain (Hyperledger Fabric) with federated learning (FL). Built on Fabric, ICBAC uses a multi-channel architecture and three smart contracts for asset management, baseline access control, and dynamic revocation. To counter insider misuse, each channel deploys an AI agent that monitors activity and dynamically restricts access for anomalies. Federated learning allows these agents to collaboratively improve detection models without sharing raw data.
  For heterogeneous, competitive environments, ICBAC introduces a game-theoretic client selection mechanism using hedonic coalition formation. This enables supply chains to form stable, strategy-proof FL coalitions via preference-based selection without disclosing sensitive criteria. Extensive experiments on a Fabric testbed with a real-world dataset show ICBAC achieves blockchain performance comparable to static frameworks and provides effective anomaly detection under IID and non-IID data with zero raw-data sharing. ICBAC thus offers a practical, scalable solution for dynamic, privacy-preserving access control in decentralized supply chains.

### ACORN-IDS: Adaptive Continual Novelty Detection for Intrusion Detection Systems

**Authors:** Sean Fuhrman, Onat Gungor, Tajana Rosing
[arXiv](http://arxiv.org/abs/2602.07291v1) · [PDF](https://arxiv.org/pdf/2602.07291v1)

Intrusion Detection Systems (IDS) must maintain reliable detection performance under rapidly evolving benign traffic patterns and the continual emergence of cyberattacks, including zero-day threats with no labeled data available. However, most machine learning-based IDS approaches either assume static data distributions or rely on labeled attack samples, substantially limiting their applicability in real-world deployments. This setting naturally motivates continual novelty detection, which enables IDS models to incrementally adapt to non-stationary data streams without labeled attack data. In this work, we introduce ACORN-IDS, an adaptive continual novelty detection framework that learns exclusively from normal data while exploiting the inherent structure of an evolving unlabeled data stream. ACORN-IDS integrates a continual feature extractor, trained using reconstruction and metric learning objectives with clustering-based pseudo-labels, alongside a PCA-based reconstruction module for anomaly scoring. This design allows ACORN-IDS to continuously adapt to distributional shifts in both benign and malicious traffic. We conduct an extensive evaluation of ACORN-IDS on five realistic intrusion datasets under two continual learning scenarios: (i) Evolving Attacks and (ii) Evolving Normal and Attack Distributions. ACORN-IDS achieves, on average, a 62% improvement in F1-score and a 58% improvement in zero-day attack detection over the state-of-the-art unsupervised continual learning baseline. It also outperforms existing state-of-the-art novelty detection approaches while exhibiting near-zero forgetting and imposing minimal inference overhead. These results demonstrate that ACORN-IDS offers a practical, label-efficient solution for building adaptive and robust IDS in dynamic, real-world environments. We plan to release the code upon acceptance.

### AirCatch: Effectively tracing advanced tag-based trackers

**Authors:** Abhishek Kumar Mishra, Swadeep, Guevara Noubir, Mathieu Cunche
[arXiv](http://arxiv.org/abs/2602.07656v1) · [PDF](https://arxiv.org/pdf/2602.07656v1)

Tag-based tracking ecosystems help users locate lost items, but can be leveraged for unwanted tracking and stalking. Existing protocol-driven defenses and prior academic solutions largely assume stable identifiers or predictable beaconing. However, identifier-based defenses fundamentally break down against advanced rogue trackers that aggressively rotate identifiers. We present AirCatch, a passive detection system that exploits a physical-layer constraint: while logical identifiers can change arbitrarily fast, the transmitter's analog imprint remains stable and reappears as a compact and persistently occupied region in Carrier Frequency Offset (CFO) feature space. AirCatch advances the state of the art along three axes: (i) a novel, modulation-aware CFO fingerprint that augments packet-level CFO with content-independent CFO components that amplify device distinctiveness; (ii) a new tracking detection algorithm based on high core density and persistence that is robust to contamination and evasion through per-identifier segmentation; and (iii) an ultra-low-cost receiver, an approximately 10 dollar BLE SDR named BlePhasyr, built from commodity components, that makes RF fingerprinting based detection practical in resource-constrained deployments. We evaluate AirCatch across Apple, Google, Tile, and Samsung tag families in multi-hour captures, systematically stress-test evasion using a scenario generator over a grid of transmission and rotation periods, and validate in diverse real-world mobility traces including home and office commutes, public transport, car travel, and airport journeys while sweeping background tag density. Across these stress tests, AirCatch achieves no false positives and early detection over a wide range of adversarial configurations and environments, degrading gracefully only in extreme low-rate regimes that also reduce attacker utility.

### AlertBERT: A noise-robust alert grouping framework for simultaneous cyber attacks

**Authors:** Lukas Karner, Max Landauer, Markus Wurzenberger, Florian Skopik
[arXiv](http://arxiv.org/abs/2602.06534v1) · [PDF](https://arxiv.org/pdf/2602.06534v1)

Automated detection of cyber attacks is a critical capability to counteract the growing volume and sophistication of cyber attacks. However, the high numbers of security alerts issued by intrusion detection systems lead to alert fatigue among analysts working in security operations centres (SOC), which in turn causes slow reaction time and incorrect decision making. Alert grouping, which refers to clustering of security alerts according to their underlying causes, can significantly reduce the number of distinct items analysts have to consider. Unfortunately, conventional time-based alert grouping solutions are unsuitable for large scale computer networks characterised by high levels of false positive alerts and simultaneously occurring attacks. To address these limitations, we propose AlertBERT, a self-supervised framework designed to group alerts from isolated or concurrent attacks in noisy environments. Thereby, our open-source implementation of AlertBERT leverages masked-language-models and density-based clustering to support both real-time or forensic operation. To evaluate our framework, we further introduce a novel data augmentation method that enables flexible control over noise levels and simulates concurrent attack occurrences. Based on the data sets generated through this method, we demonstrate that AlertBERT consistently outperforms conventional time-based grouping techniques, achieving superior accuracy in identifying correct alert groups.

### Empirical Analysis of Adversarial Robustness and Explainability Drift in Cybersecurity Classifiers

**Authors:** Mona Rajhans, Vishal Khawarey
[arXiv](http://arxiv.org/abs/2602.06395v1) · [PDF](https://arxiv.org/pdf/2602.06395v1)

Machine learning (ML) models are increasingly deployed in cybersecurity applications such as phishing detection and network intrusion prevention. However, these models remain vulnerable to adversarial perturbations small, deliberate input modifications that can degrade detection accuracy and compromise interpretability. This paper presents an empirical study of adversarial robustness and explainability drift across two cybersecurity domains phishing URL classification and network intrusion detection. We evaluate the impact of L (infinity) bounded Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) perturbations on model accuracy and introduce a quantitative metric, the Robustness Index (RI), defined as the area under the accuracy perturbation curve. Gradient based feature sensitivity and SHAP based attribution drift analyses reveal which input features are most susceptible to adversarial manipulation. Experiments on the Phishing Websites and UNSW NB15 datasets show consistent robustness trends, with adversarial training improving RI by up to 9 percent while maintaining clean-data accuracy. These findings highlight the coupling between robustness and interpretability degradation and underscore the importance of quantitative evaluation in the design of trustworthy, AI-driven cybersecurity systems.

### Interpreting Manifolds and Graph Neural Embeddings from Internet of Things Traffic Flows

**Authors:** Enrique Feito-Casares, Francisco M. Melgarejo-Meseguer, Elena Casiraghi, Giorgio Valentini, José-Luis Rojo-Álvarez
[arXiv](http://arxiv.org/abs/2602.05817v2) · [PDF](https://arxiv.org/pdf/2602.05817v2)

The rapid expansion of Internet of Things (IoT) ecosystems has led to increasingly complex and heterogeneous network topologies. Traditional network monitoring and visualization tools rely on aggregated metrics or static representations, which fail to capture the evolving relationships and structural dependencies between devices. Although Graph Neural Networks (GNNs) offer a powerful way to learn from relational data, their internal representations often remain opaque and difficult to interpret for security-critical operations. Consequently, this work introduces an interpretable pipeline that generates directly visualizable low-dimensional representations by mapping high-dimensional embeddings onto a latent manifold. This projection enables the interpretable monitoring and interoperability of evolving network states, while integrated feature attribution techniques decode the specific characteristics shaping the manifold structure. The framework achieves a classification F1-score of 0.830 for intrusion detection while also highlighting phenomena such as concept drift. Ultimately, the presented approach bridges the gap between high-dimensional GNN embeddings and human-understandable network behavior, offering new insights for network administrators and security analysts.

### Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment

**Authors:** Abdelkader El Mahdaouy, Issam Ait Yahia, Soufiane Oualil, Ismail Berrada
[arXiv](http://arxiv.org/abs/2602.05594v1) · [PDF](https://arxiv.org/pdf/2602.05594v1)

Network Intrusion Detection Systems (NIDS) have progressively shifted from signature-based techniques toward machine learning and, more recently, deep learning methods. Meanwhile, the widespread adoption of encryption has reduced payload visibility, weakening inspection pipelines that depend on plaintext content and increasing reliance on flow-level telemetry such as NetFlow and IPFIX. Many current learning-based detectors still frame intrusion detection as per-flow classification, implicitly treating each flow record as an independent sample. This assumption is often violated in realistic attack campaigns, where evidence is distributed across multiple flows and hosts, spanning minutes to days through staged execution, beaconing, lateral movement, and exfiltration. This paper synthesizes recent research on context-aware deep learning for flow-based intrusion detection. We organize existing methods into a four-dimensional taxonomy covering temporal context, graph or relational context, multimodal context, and multi-resolution context. Beyond modeling, we emphasize rigorous evaluation and operational realism. We review common failure modes that can inflate reported results, including temporal leakage, data splitting, dataset design flaws, limited dataset diversity, and weak cross-dataset generalization. We also analyze practical constraints that shape deployability, such as streaming state management, memory growth, latency budgets, and model compression choices. Overall, the literature suggests that context can meaningfully improve detection when attacks induce measurable temporal or relational structure, but the magnitude and reliability of these gains depend strongly on rigorous, causal evaluation and on datasets that capture realistic diversity.

---

## Machine Learning

*Papers in Machine Learning*

### Interpretable Graph-Level Anomaly Detection via Contrast with Normal Prototypes

**Authors:** Qiuran Zhao, Kai Ming Ting, Xinpeng Li
[arXiv](http://arxiv.org/abs/2602.10708v1) · [PDF](https://arxiv.org/pdf/2602.10708v1)

The task of graph-level anomaly detection (GLAD) is to identify anomalous graphs that deviate significantly from the majority of graphs in a dataset. While deep GLAD methods have shown promising performance, their black-box nature limits their reliability and deployment in real-world applications. Although some recent methods have made attempts to provide explanations for anomaly detection results, they either provide explanations without referencing normal graphs, or rely on abstract latent vectors as prototypes rather than concrete graphs from the dataset. To address these limitations, we propose Prototype-based Graph-Level Anomaly Detection (ProtoGLAD), an interpretable unsupervised framework that provides explanation for each detected anomaly by explicitly contrasting with its nearest normal prototype graph. It employs a point-set kernel to iteratively discover multiple normal prototype graphs and their associated clusters from the dataset, then identifying graphs distant from all discovered normal clusters as anomalies. Extensive experiments on multiple real-world datasets demonstrate that ProtoGLAD achieves competitive anomaly detection performance compared to state-of-the-art GLAD methods while providing better human-interpretable prototype-based explanations.

### A Dual-Stream Physics-Augmented Unsupervised Architecture for Runtime Embedded Vehicle Health Monitoring

**Authors:** Enzo Nicolas Spotorno, Antonio Augusto Medeiros Frohlich
[arXiv](http://arxiv.org/abs/2602.10432v1) · [PDF](https://arxiv.org/pdf/2602.10432v1)

Runtime quantification of vehicle operational intensity is essential for predictive maintenance and condition monitoring in commercial and heavy-duty fleets. Traditional metrics like mileage fail to capture mechanical burden, while unsupervised deep learning models detect statistical anomalies, typically transient surface shocks, but often conflate statistical stability with mechanical rest. We identify this as a critical blind spot: high-load steady states, such as hill climbing with heavy payloads, appear statistically normal yet impose significant drivetrain fatigue. To resolve this, we propose a Dual-Stream Architecture that fuses unsupervised learning for surface anomaly detection with macroscopic physics proxies for cumulative load estimation. This approach leverages low-frequency sensor data to generate a multi-dimensional health vector, distinguishing between dynamic hazards and sustained mechanical effort. Validated on a RISC-V embedded platform, the architecture demonstrates low computational overhead, enabling comprehensive, edge-based health monitoring on resource-constrained ECUs without the latency or bandwidth costs of cloud-based monitoring.

### Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings

**Authors:** Alexander Fertig, Karthikeyan Chandra Sekaran, Lakshman Balasubramanian, Michael Botsch
[arXiv](http://arxiv.org/abs/2602.09985v1) · [PDF](https://arxiv.org/pdf/2602.09985v1)

As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection methods, in order to identify anomalies within object state representations. This framework is particularly useful for applications in real-world environments, where new or unknown anomalies may occur during operation for which there are no labels available. Experiments performed on the publicly available, real-world nuScenes dataset illustrate the framework's capabilities.

### Contextual and Seasonal LSTMs for Time Series Anomaly Detection

**Authors:** Lingpei Zhang, Qingming Li, Yong Yang, Jiahao Chen, Rui Zeng, Chenyang Lyu, Shouling Ji
[arXiv](http://arxiv.org/abs/2602.09690v1) · [PDF](https://arxiv.org/pdf/2602.09690v1)

Univariate time series (UTS), where each timestamp records a single variable, serve as crucial indicators in web systems and cloud servers. Anomaly detection in UTS plays an essential role in both data mining and system reliability management. However, existing reconstruction-based and prediction-based methods struggle to capture certain subtle anomalies, particularly small point anomalies and slowly rising anomalies. To address these challenges, we propose a novel prediction-based framework named Contextual and Seasonal LSTMs (CS-LSTMs). CS-LSTMs are built upon a noise decomposition strategy and jointly leverage contextual dependencies and seasonal patterns, thereby strengthening the detection of subtle anomalies. By integrating both time-domain and frequency-domain representations, CS-LSTMs achieve more accurate modeling of periodic trends and anomaly localization. Extensive evaluations on public benchmark datasets demonstrate that CS-LSTMs consistently outperform state-of-the-art methods, highlighting their effectiveness and practical value in robust time series anomaly detection.

### Why the Counterintuitive Phenomenon of Likelihood Rarely Appears in Tabular Anomaly Detection with Deep Generative Models?

**Authors:** Donghwan Kim, Junghun Phee, Hyunsoo Yoon
[arXiv](http://arxiv.org/abs/2602.09593v1) · [PDF](https://arxiv.org/pdf/2602.09593v1)

Deep generative models with tractable and analytically computable likelihoods, exemplified by normalizing flows, offer an effective basis for anomaly detection through likelihood-based scoring. We demonstrate that, unlike in the image domain where deep generative models frequently assign higher likelihoods to anomalous data, such counterintuitive behavior occurs far less often in tabular settings. We first introduce a domain-agnostic formulation that enables consistent detection and evaluation of the counterintuitive phenomenon, addressing the absence of precise definition. Through extensive experiments on 47 tabular datasets and 10 CV/NLP embedding datasets in ADBench, benchmarked against 13 baseline models, we demonstrate that the phenomenon, as defined, is consistently rare in general tabular data. We further investigate this phenomenon from both theoretical and empirical perspectives, focusing on the roles of data dimensionality and difference in feature correlation. Our results suggest that likelihood-only detection with normalizing flows offers a practical and reliable approach for anomaly detection in tabular domains.

### PlugSI: Plug-and-Play Test-Time Graph Adaptation for Spatial Interpolation

**Authors:** Xuhang Wu, Zhuoxuan Liang, Wei Li, Xiaohua Jia, Sumi Helal
[arXiv](http://arxiv.org/abs/2602.09824v1) · [PDF](https://arxiv.org/pdf/2602.09824v1)

With the rapid advancement of IoT and edge computing, sensor networks have become indispensable, driving the need for large-scale sensor deployment. However, the high deployment cost hinders their scalability. To tackle the issues, Spatial Interpolation (SI) introduces virtual sensors to infer readings from observed sensors, leveraging graph structure. However, current graph-based SI methods rely on pre-trained models, lack adaptation to larger and unseen graphs at test-time, and overlook test data utilization. To address these issues, we propose PlugSI, a plug-and-play framework that refines test-time graph through two key innovations. First, we design an Unknown Topology Adapter (UTA) that adapts to the new graph structure of each small-batch at test-time, enhancing the generalization of SI pre-trained models. Second, we introduce a Temporal Balance Adapter (TBA) that maintains a stable historical consensus to guide UTA adaptation and prevent drifting caused by noise in the current batch. Empirically, extensive experiments demonstrate PlugSI can be seamlessly integrated into existing graph-based SI methods and provide significant improvement (e.g., a 10.81% reduction in MAE).

### DynamiQ: Accelerating Gradient Synchronization using Compressed Multi-hop All-reduce

**Authors:** Wenchen Han, Shay Vargaftik, Michael Mitzenmacher, Ran Ben Basat
[arXiv](http://arxiv.org/abs/2602.08923v1) · [PDF](https://arxiv.org/pdf/2602.08923v1)

Multi-hop all-reduce is the de facto backbone of large model training. As the training scale increases, the network often becomes a bottleneck, motivating reducing the volume of transmitted data. Accordingly, recent systems demonstrated significant acceleration of the training process using gradient quantization. However, these systems are not optimized for multi-hop aggregation, where entries are partially summed multiple times along their aggregation topology.
  This paper presents DynamiQ, a quantization framework that bridges the gap between quantization best practices and multi-hop aggregation. DynamiQ introduces novel techniques to better represent partial sums, co-designed with a decompress-accumulate-recompress fused kernel to facilitate fast execution.
  We extended PyTorch DDP to support DynamiQ over NCCL P2P, and across different LLMs, tasks, and scales, we demonstrate consistent improvement of up to 34.2% over the best among state-of-the-art methods such as Omni-Reduce, THC, and emerging standards such as MXFP4, MXFP6, and MXFP8. Further, DynamiQ is the only evaluated method that consistently reaches near-baseline accuracy (e.g., 99.9% of the BF16 baseline) and does so while significantly accelerating the training.

### RIFLE: Robust Distillation-based FL for Deep Model Deployment on Resource-Constrained IoT Networks

**Authors:** Pouria Arefijamal, Mahdi Ahmadlou, Bardia Safaei, Jörg Henkel
[arXiv](http://arxiv.org/abs/2602.08446v1) · [PDF](https://arxiv.org/pdf/2602.08446v1)

Federated learning (FL) is a decentralized learning paradigm widely adopted in resource-constrained Internet of Things (IoT) environments. These devices, typically relying on TinyML models, collaboratively train global models by sharing gradients with a central server while preserving data privacy. However, as data heterogeneity and task complexity increase, TinyML models often become insufficient to capture intricate patterns, especially under extreme non-IID (non-independent and identically distributed) conditions. Moreover, ensuring robustness against malicious clients and poisoned updates remains a major challenge. Accordingly, this paper introduces RIFLE - a Robust, distillation-based Federated Learning framework that replaces gradient sharing with logit-based knowledge transfer. By leveraging a knowledge distillation aggregation scheme, RIFLE enables the training of deep models such as VGG-19 and Resnet18 within constrained IoT systems. Furthermore, a Kullback-Leibler (KL) divergence-based validation mechanism quantifies the reliability of client updates without exposing raw data, achieving high trust and privacy preservation simultaneously. Experiments on three benchmark datasets (MNIST, CIFAR-10, and CIFAR-100) under heterogeneous non-IID conditions demonstrate that RIFLE reduces false-positive detections by up to 87.5%, enhances poisoning attack mitigation by 62.5%, and achieves up to 28.3% higher accuracy compared to conventional federated learning baselines within only 10 rounds. Notably, RIFLE reduces VGG19 training time from over 600 days to just 1.39 hours on typical IoT devices (0.3 GFLOPS), making deep learning practical in resource-constrained networks.

### Importance inversion transfer identifies shared principles for cross-domain learning

**Authors:** Daniele Caligiore
[arXiv](http://arxiv.org/abs/2602.09116v2) · [PDF](https://arxiv.org/pdf/2602.09116v2)

The capacity to transfer knowledge across scientific domains relies on shared organizational principles. However, existing transfer-learning methodologies often fail to bridge radically heterogeneous systems, particularly under severe data scarcity or stochastic noise. This study formalizes Explainable Cross-Domain Transfer Learning (X-CDTL), a framework unifying network science and explainable artificial intelligence to identify structural invariants that generalize across biological, linguistic, molecular, and social networks. By introducing the Importance Inversion Transfer (IIT) mechanism, the framework prioritizes domain-invariant structural anchors over idiosyncratic, highly discriminative features. In anomaly detection tasks, models guided by these principles achieve significant performance gains - exhibiting a 56% relative improvement in decision stability under extreme noise - over traditional baselines. These results provide evidence for a shared organizational signature across heterogeneous domains, establishing a principled paradigm for cross-disciplinary knowledge propagation. By shifting from opaque latent representations to explicit structural laws, this work advances machine learning as a robust engine for scientific discovery.

### AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection

**Authors:** Junru Zhang, Lang Feng, Haoran Shi, Xu Guo, Han Yu, Yabo Dong, Duanqing Xu
[arXiv](http://arxiv.org/abs/2602.08868v1) · [PDF](https://arxiv.org/pdf/2602.08868v1)

Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions.

### LEFT: Learnable Fusion of Tri-view Tokens for Unsupervised Time Series Anomaly Detection

**Authors:** Dezheng Wang, Tong Chen, Guansong Pang, Congyan Chen, Shihua Li, Hongzhi Yin
[arXiv](http://arxiv.org/abs/2602.08638v1) · [PDF](https://arxiv.org/pdf/2602.08638v1)

As a fundamental data mining task, unsupervised time series anomaly detection (TSAD) aims to build a model for identifying abnormal timestamps without assuming the availability of annotations. A key challenge in unsupervised TSAD is that many anomalies are too subtle to exhibit detectable deviation in any single view (e.g., time domain), and instead manifest as inconsistencies across multiple views like time, frequency, and a mixture of resolutions. However, most cross-view methods rely on feature or score fusion and do not enforce analysis-synthesis consistency, meaning the frequency branch is not required to reconstruct the time signal through an inverse transform, and vice versa. In this paper, we present Learnable Fusion of Tri-view Tokens (LEFT), a unified unsupervised TSAD framework that models anomalies as inconsistencies across complementary representations. LEFT learns feature tokens from three views of the same input time series: frequency-domain tokens that embed periodicity information, time-domain tokens that capture local dynamics, and multi-scale tokens that learns abnormal patterns at varying time series granularities. By learning a set of adaptive Nyquist-constrained spectral filters, the original time series is rescaled into multiple resolutions and then encoded, allowing these multi-scale tokens to complement the extracted frequency- and time-domain information. When generating the fused representation, we introduce a novel objective that reconstructs fine-grained targets from coarser multi-scale structure, and put forward an innovative time-frequency cycle consistency constraint to explicitly regularize cross-view agreement. Experiments on real-world benchmarks show that LEFT yields the best detection accuracy against SOTA baselines, while achieving a 5x reduction on FLOPs and 8x speed-up for training.

### Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization

**Authors:** Charalampos Shimillas, Kleanthis Malialis, Konstantinos Fokianos, Marios M. Polycarpou
[arXiv](http://arxiv.org/abs/2602.08467v1) · [PDF](https://arxiv.org/pdf/2602.08467v1)

Multivariate time series (MTS) anomaly diagnosis, which encompasses both anomaly detection and localization, is critical for the safety and reliability of complex, large-scale real-world systems. The vast majority of existing anomaly diagnosis methods offer limited theoretical insights, especially for anomaly localization, which is a vital but largely unexplored area. The aim of this contribution is to study the learning process of a Transformer when applied to MTS by revealing connections to statistical time series methods. Based on these theoretical insights, we propose the Attention Low-Rank Transformer (ALoRa-T) model, which applies low-rank regularization to self-attention, and we introduce the Attention Low-Rank score, effectively capturing the temporal characteristics of anomalies. Finally, to enable anomaly localization, we propose the ALoRa-Loc method, a novel approach that associates anomalies to specific variables by quantifying interrelationships among time series. Extensive experiments and real data analysis, show that the proposed methodology significantly outperforms state-of-the-art methods in both detection and localization tasks.

### Towards Efficient Large Language Reasoning Models via Extreme-Ratio Chain-of-Thought Compression

**Authors:** Yuntian Tang, Bohan Jia, Wenxuan Huang, Lianyue Zhang, Jiao Xie, Wenxi Li, Wei Li, Jie Hu, Xinghao Chen, Rongrong Ji, Shaohui Lin
[arXiv](http://arxiv.org/abs/2602.08324v1) · [PDF](https://arxiv.org/pdf/2602.08324v1)

Chain-of-Thought (CoT) reasoning successfully enhances the reasoning capabilities of Large Language Models (LLMs), yet it incurs substantial computational overhead for inference. Existing CoT compression methods often suffer from a critical loss of logical fidelity at high compression ratios, resulting in significant performance degradation. To achieve high-fidelity, fast reasoning, we propose a novel EXTreme-RAtio Chain-of-Thought Compression framework, termed Extra-CoT, which aggressively reduces the token budget while preserving answer accuracy. To generate reliable, high-fidelity supervision, we first train a dedicated semantically-preserved compressor on mathematical CoT data with fine-grained annotations. An LLM is then fine-tuned on these compressed pairs via a mixed-ratio supervised fine-tuning (SFT), teaching it to follow a spectrum of compression budgets and providing a stable initialization for reinforcement learning (RL). We further propose Constrained and Hierarchical Ratio Policy Optimization (CHRPO) to explicitly incentivize question-solving ability under lower budgets by a hierarchical reward. Experiments on three mathematical reasoning benchmarks show the superiority of Extra-CoT. For example, on MATH-500 using Qwen3-1.7B, Extra-CoT achieves over 73\% token reduction with an accuracy improvement of 0.6\%, significantly outperforming state-of-the-art (SOTA) methods.

### Online Bayesian Imbalanced Learning with Bregman-Calibrated Deep Networks

**Authors:** Zahir Alsulaimawi
[arXiv](http://arxiv.org/abs/2602.08128v1) · [PDF](https://arxiv.org/pdf/2602.08128v1)

Class imbalance remains a fundamental challenge in machine learning, where standard classifiers exhibit severe performance degradation in minority classes. Although existing approaches address imbalance through resampling or cost-sensitive learning during training, they require retraining or access to labeled target data when class distributions shift at deployment time, a common occurrence in real-world applications such as fraud detection, medical diagnosis, and anomaly detection. We present \textit{Online Bayesian Imbalanced Learning} (OBIL), a principled framework that decouples likelihood-ratio estimation from class-prior assumptions, enabling real-time adaptation to distribution shifts without model retraining. Our approach builds on the established connection between Bregman divergences and proper scoring rules to show that deep networks trained with such losses produce posterior probability estimates from which prior-invariant likelihood ratios can be extracted. We prove that these likelihood-ratio estimates remain valid under arbitrary changes in class priors and cost structures, requiring only a threshold adjustment for optimal Bayes decisions. We derive finite-sample regret bounds demonstrating that OBIL achieves $O(\sqrt{T \log T})$ regret against an oracle with perfect prior knowledge. Extensive experiments on benchmark datasets and medical diagnosis benchmarks under simulated deployment shifts demonstrate that OBIL maintains robust performance under severe distribution shifts, outperforming state-of-the-art methods in F1 Score when test distributions deviate significantly from the training conditions.

### CausalTAD: Injecting Causal Knowledge into Large Language Models for Tabular Anomaly Detection

**Authors:** Ruiqi Wang, Ruikang Liu, Runyu Chen, Haoxiang Suo, Zhiyi Peng, Zhuo Tang, Changjian Chen
[arXiv](http://arxiv.org/abs/2602.07798v1) · [PDF](https://arxiv.org/pdf/2602.07798v1)

Detecting anomalies in tabular data is critical for many real-world applications, such as credit card fraud detection. With the rapid advancements in large language models (LLMs), state-of-the-art performance in tabular anomaly detection has been achieved by converting tabular data into text and fine-tuning LLMs. However, these methods randomly order columns during conversion, without considering the causal relationships between them, which is crucial for accurately detecting anomalies. In this paper, we present CausalTaD, a method that injects causal knowledge into LLMs for tabular anomaly detection. We first identify the causal relationships between columns and reorder them to align with these causal relationships. This reordering can be modeled as a linear ordering problem. Since each column contributes differently to the causal relationships, we further propose a reweighting strategy to assign different weights to different columns to enhance this effect. Experiments across more than 30 datasets demonstrate that our method consistently outperforms the current state-of-the-art methods. The code for CausalTAD is available at https://github.com/350234/CausalTAD.

### Beyond Pooling: Matching for Robust Generalization under Data Heterogeneity

**Authors:** Ayush Roy, Rudrasis Chakraborty, Lav Varshney, Vishnu Suresh Lokhande
[arXiv](http://arxiv.org/abs/2602.07154v1) · [PDF](https://arxiv.org/pdf/2602.07154v1)

Pooling heterogeneous datasets across domains is a common strategy in representation learning, but naive pooling can amplify distributional asymmetries and yield biased estimators, especially in settings where zero-shot generalization is required. We propose a matching framework that selects samples relative to an adaptive centroid and iteratively refines the representation distribution. The double robustness and the propensity score matching for the inclusion of data domains make matching more robust than naive pooling and uniform subsampling by filtering out the confounding domains (the main cause of heterogeneity). Theoretical and empirical analyses show that, unlike naive pooling or uniform subsampling, matching achieves better results under asymmetric meta-distributions, which are also extended to non-Gaussian and multimodal real-world settings. Most importantly, we show that these improvements translate to zero-shot medical anomaly detection, one of the extreme forms of data heterogeneity and asymmetry. The code is available on https://github.com/AyushRoy2001/Beyond-Pooling.

### Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts

**Authors:** Xinyu Zhao, Qingyun Sun, Jiayi Luo, Xingcheng Fu, Jianxin Li
[arXiv](http://arxiv.org/abs/2602.06859v2) · [PDF](https://arxiv.org/pdf/2602.06859v2)

Graph Anomaly Detection (GAD) aims to identify irregular patterns in graph data, and recent works have explored zero-shot generalist GAD to enable generalization to unseen graph datasets. However, existing zero-shot GAD methods largely ignore intrinsic geometric differences across diverse anomaly patterns, substantially limiting their cross-domain generalization. In this work, we reveal that anomaly detectability is highly dependent on the underlying geometric properties and that embedding graphs from different domains into a single static curvature space can distort the structural signatures of anomalies. To address the challenge that a single curvature space cannot capture geometry-dependent graph anomaly patterns, we propose GAD-MoRE, a novel framework for zero-shot Generalizable Graph Anomaly Detection with a Mixture of Riemannian Experts architecture. Specifically, to ensure that each anomaly pattern is modeled in the Riemannian space where it is most detectable, GAD-MoRE employs a set of specialized Riemannian expert networks, each operating in a distinct curvature space. To align raw node features with curvature-specific anomaly characteristics, we introduce an anomaly-aware multi-curvature feature alignment module that projects inputs into parallel Riemannian spaces, enabling the capture of diverse geometric characteristics. Finally, to facilitate better generalization beyond seen patterns, we design a memory-based dynamic router that adaptively assigns each input to the most compatible expert based on historical reconstruction performance on similar anomalies. Extensive experiments in the zero-shot setting demonstrate that GAD-MoRE significantly outperforms state-of-the-art generalist GAD baselines, and even surpasses strong competitors that are few-shot fine-tuned with labeled data from the target domain.

### Calibrating Tabular Anomaly Detection via Optimal Transport

**Authors:** Hangting Ye, He Zhao. Wei Fan, Xiaozhuang Song, Dandan Guo, Yi Chang, Hongyuan Zha
[arXiv](http://arxiv.org/abs/2602.06810v1) · [PDF](https://arxiv.org/pdf/2602.06810v1)

Tabular anomaly detection (TAD) remains challenging due to the heterogeneity of tabular data: features lack natural relationships, vary widely in distribution and scale, and exhibit diverse types. Consequently, each TAD method makes implicit assumptions about anomaly patterns that work well on some datasets but fail on others, and no method consistently outperforms across diverse scenarios. We present CTAD (Calibrating Tabular Anomaly Detection), a model-agnostic post-processing framework that enhances any existing TAD detector through sample-specific calibration. Our approach characterizes normal data via two complementary distributions, i.e., an empirical distribution from random sampling and a structural distribution from K-means centroids, and measures how adding a test sample disrupts their compatibility using Optimal Transport (OT) distance. Normal samples maintain low disruption while anomalies cause high disruption, providing a calibration signal to amplify detection. We prove that OT distance has a lower bound proportional to the test sample's distance from centroids, and establish that anomalies systematically receive higher calibration scores than normals in expectation, explaining why the method generalizes across datasets. Extensive experiments on 34 diverse tabular datasets with 7 representative detectors spanning all major TAD categories (density estimation, classification, reconstruction, and isolation-based methods) demonstrate that CTAD consistently improves performance with statistical significance. Remarkably, CTAD enhances even state-of-the-art deep learning methods and shows robust performance across diverse hyperparameter settings, requiring no additional tuning for practical deployment.

### Empowering Time Series Analysis with Large-Scale Multimodal Pretraining

**Authors:** Peng Chen, Siyuan Wang, Shiyan Hu, Xingjian Wu, Yang Shu, Zhongwen Rao, Meng Wang, Yijie Li, Bin Yang, Chenjuan Guo
[arXiv](http://arxiv.org/abs/2602.05646v1) · [PDF](https://arxiv.org/pdf/2602.05646v1)

While existing time series foundation models primarily rely on large-scale unimodal pretraining, they lack complementary modalities to enhance time series understanding. Building multimodal foundation models is a natural next step, but it faces key challenges: 1) lack of a unified multimodal pretraining paradigm and large-scale multimodal corpora for time series analysis; 2) how to effectively integrate heterogeneous modalities and enhance model generalization. To address these challenges, we take an early step toward multimodal foundation models for time series analysis. We first propose a multimodal pretraining paradigm that leverages time series with endogenous modalities (derived images and text) and exogenous knowledge (real-world news), providing a comprehensive multi-view perspective for time series analysis. To support this, we develop an automated data construction pipeline to curate MM-TS, the first large-scale multimodal time series dataset spanning six domains, with up to one billion points. Then we propose HORAI, a frequency-enhanced multimodal foundation model. It integrates two core components: the Frequency-enhanced Cross-Modality Encoder and the Time-Frequency Decoder, designed to effectively fuse multimodal features and enhance model generalization across modalities and domains. After pretraining on MM-TS, HORAI achieves state-of-the-art zero-shot performance on time series forecasting and anomaly detection tasks, demonstrating strong generalization.

### Joint Embedding Variational Bayes

**Authors:** Amin Oji, Paul Fieguth
[arXiv](http://arxiv.org/abs/2602.05639v1) · [PDF](https://arxiv.org/pdf/2602.05639v1)

We introduce Variational Joint Embedding (VJE), a framework that synthesizes joint embedding and variational inference to enable self-supervised learning of probabilistic representations in a reconstruction-free, non-contrastive setting. Compared to energy-based predictive objectives that optimize pointwise discrepancies, VJE maximizes a symmetric conditional evidence lower bound (ELBO) for a latent-variable model defined directly on encoder embeddings. We instantiate the conditional likelihood with a heavy-tailed Student-$t$ model using a polar decomposition that explicitly decouples directional and radial factors to prevent norm-induced instabilities during training. VJE employs an amortized inference network to parameterize a diagonal Gaussian variational posterior whose feature-wise variances are shared with the likelihood scale to capture anisotropic uncertainty without auxiliary projection heads. Across ImageNet-1K, CIFAR-10/100, and STL-10, VJE achieves performance comparable to standard non-contrastive baselines under linear and k-NN evaluation. We further validate these probabilistic semantics through one-class CIFAR-10 anomaly detection, where likelihood-based scoring under the proposed model outperforms comparable self-supervised baselines.

### Balanced Anomaly-guided Ego-graph Diffusion Model for Inductive Graph Anomaly Detection

**Authors:** Chunyu Wei, Siyuan He, Yu Wang, Yueguo Chen, Yunhai Wang, Bing Bai, Yidong Zhang, Yong Xie, Shunming Zhang, Fei Wang
[arXiv](http://arxiv.org/abs/2602.05232v1) · [PDF](https://arxiv.org/pdf/2602.05232v1)

Graph anomaly detection (GAD) is crucial in applications like fraud detection and cybersecurity. Despite recent advancements using graph neural networks (GNNs), two major challenges persist. At the model level, most methods adopt a transductive learning paradigm, which assumes static graph structures, making them unsuitable for dynamic, evolving networks. At the data level, the extreme class imbalance, where anomalous nodes are rare, leads to biased models that fail to generalize to unseen anomalies. These challenges are interdependent: static transductive frameworks limit effective data augmentation, while imbalance exacerbates model distortion in inductive learning settings. To address these challenges, we propose a novel data-centric framework that integrates dynamic graph modeling with balanced anomaly synthesis. Our framework features: (1) a discrete ego-graph diffusion model, which captures the local topology of anomalies to generate ego-graphs aligned with anomalous structural distribution, and (2) a curriculum anomaly augmentation mechanism, which dynamically adjusts synthetic data generation during training, focusing on underrepresented anomaly patterns to improve detection and generalization. Experiments on five datasets demonstrate that the effectiveness of our framework.

---

## cs.HC

*Papers in cs.HC*

### Towards Affordable, Non-Invasive Real-Time Hypoglycemia Detection Using Wearable Sensor Signals

**Authors:** Lawrence Obiuwevwi, Krzysztof J. Rechowicz, Vikas Ashok, Sampath Jayarathna
[arXiv](http://arxiv.org/abs/2602.10407v1) · [PDF](https://arxiv.org/pdf/2602.10407v1)

Accurately detecting hypoglycemia without invasive glucose sensors remains a critical challenge in diabetes management, particularly in regions where continuous glucose monitoring (CGM) is prohibitively expensive or clinically inaccessible. This extended study introduces a comprehensive, multimodal physiological framework for non-invasive hypoglycemia detection using wearable sensor signals. Unlike prior work limited to single-signal analysis, this chapter evaluates three physiological modalities, galvanic skin response (GSR), heart rate (HR), and their combined fusion, using the OhioT1DM 2018 dataset. We develop an end-to-end pipeline that integrates advanced preprocessing, temporal windowing, handcrafted and sequence-based feature extraction, early and late fusion strategies, and a broad spectrum of machine learning and deep temporal models, including CNNs, LSTMs, GRUs, and TCNs. Our results demonstrate that physiological signals exhibit distinct autonomic patterns preceding hypoglycemia and that combining GSR with HR consistently enhances detection sensitivity and stability compared to single-signal models. Multimodal deep learning architectures achieve the most reliable performance, particularly in recall, the most clinically urgent metric. Ablation studies further highlight the complementary contributions of each modality, strengthening the case for affordable, sensor-based glycemic monitoring. The findings show that real-time hypoglycemia detection is achievable using only inexpensive, non-invasive wearable sensors, offering a pathway toward accessible glucose monitoring in underserved communities and low-resource healthcare environments.

---

## stat.ME

*Papers in stat.ME*

### Extended Isolation Forest with feature sensitivities

**Authors:** Illia Donhauzer
[arXiv](http://arxiv.org/abs/2602.09704v1) · [PDF](https://arxiv.org/pdf/2602.09704v1)

Compared to theoretical frameworks that assume equal sensitivity to deviations in all features of data, the theory of anomaly detection allowing for variable sensitivity across features is less developed. To the best of our knowledge, this issue has not yet been addressed in the context of isolation-based methods, and this paper represents the first attempt to do so. This paper introduces an Extended Isolation Forest with feature sensitivities, which we refer to as the Anisotropic Isolation Forest (AIF). In contrast to the standard EIF, the AIF enables anomaly detection with controllable sensitivity to deviations in different features or directions in the feature space. The paper also introduces novel measures of directional sensitivity, which allow quantification of AIF's sensitivity in different directions in the feature space. These measures enable adjustment of the AIF's sensitivity to task-specific requirements. We demonstrate the performance of the algorithm by applying it to synthetic and real-world datasets. The results show that the AIF enables anomaly detection that focuses on directions in the feature space where deviations from typical behavior are more important.

---

## cs.MM

*Papers in cs.MM*

### TAROT: Towards Optimization-Driven Adaptive FEC Parameter Tuning for Video Streaming

**Authors:** Jashanjot Singh Sidhu, Aman Sahu, Abdelhak Bentaleb
[arXiv](http://arxiv.org/abs/2602.09880v1) · [PDF](https://arxiv.org/pdf/2602.09880v1)

Forward Error Correction (FEC) remains essential for protecting video streaming against packet loss, yet most real deployments still rely on static, coarse-grained configurations that cannot react to rapid shifts in loss rate, goodput, or client buffer levels. These rigid settings often create inefficiencies: unnecessary redundancy that suppresses throughput during stable periods, and insufficient protection during bursty losses, especially when shallow buffers and oversized blocks increase stall risk. To address these challenges, we present TAROT, a cross-layer, optimization-driven FEC controller that selects redundancy, block size, and symbolization on a per-segment basis. TAROT is codec-agnostic--supporting Reed-Solomon, RaptorQ, and XOR-based codes--and evaluates a pre-computed candidate set using a fine-grained scoring model. The scoring function jointly incorporates transport-layer loss and goodput, application layer buffer dynamics, and block-level timing constraints to penalize insufficient coverage, excessive overhead, and slow block completion. To enable realistic testing, we extend the SABRE simulator 1 with two new modules: a high-fidelity packet-loss generator that replays diverse multi-trace loss patterns, and a modular FEC benchmarking layer supporting arbitrary code/parameter combinations. Across Low-Latency Live (LLL) and Video-on-Demand (VoD) streaming modes, diverse network traces, and multiple ABR algorithms, TAROT reduces FEC overhead by up to 43% while improving perceptual quality by 10 VMAF units with minimal rebuffering, achieving a stronger overhead-quality balance than static FECs.

---

## cs.RO

*Papers in cs.RO*

### Diverse Skill Discovery for Quadruped Robots via Unsupervised Learning

**Authors:** Ruopeng Cui, Yifei Bi, Haojie Luo, Wei Li
[arXiv](http://arxiv.org/abs/2602.09767v1) · [PDF](https://arxiv.org/pdf/2602.09767v1)

Reinforcement learning necessitates meticulous reward shaping by specialists to elicit target behaviors, while imitation learning relies on costly task-specific data. In contrast, unsupervised skill discovery can potentially reduce these burdens by learning a diverse repertoire of useful skills driven by intrinsic motivation. However, existing methods exhibit two key limitations: they typically rely on a single policy to master a versatile repertoire of behaviors without modeling the shared structure or distinctions among them, which results in low learning efficiency; moreover, they are susceptible to reward hacking, where the reward signal increases and converges rapidly while the learned skills display insufficient actual diversity. In this work, we introduce an Orthogonal Mixture-of-Experts (OMoE) architecture that prevents diverse behaviors from collapsing into overlapping representations, enabling a single policy to master a wide spectrum of locomotion skills. In addition, we design a multi-discriminator framework in which different discriminators operate on distinct observation spaces, effectively mitigating reward hacking. We evaluated our method on the 12-DOF Unitree A1 quadruped robot, demonstrating a diverse set of locomotion skills. Our experiments demonstrate that the proposed framework boosts training efficiency and yields an 18.3\% expansion in state-space coverage compared to the baseline.

---

## Signal Processing

*Papers in Signal Processing*

### Generalizable and Robust Beam Prediction for 6G Networks: An Deep-Learning Framework with Positioning Feature Fusion

**Authors:** Yanliang Jin, Yunfan Li, Jiang Jun, Yuan Gao, Shengli Liu, Jianbo Du, Zhaohui Yang, Shugong Xu
[arXiv](http://arxiv.org/abs/2602.09685v1) · [PDF](https://arxiv.org/pdf/2602.09685v1)

Beamforming (BF) is essential for enhancing system capacity in fifth generation (5G) and beyond wireless networks, yet exhaustive beam training in ultra-massive multiple-input multiple-output (MIMO) systems incurs substantial overhead. To address this challenge, we propose a deep learning based framework that leverages position-aware features to improve beam prediction accuracy while reducing training costs. The proposed approach uses spatial coordinate labels to supervise a position extraction branch and integrates the resulting representations with beam-domain features through a feature fusion module. A dual-branch RegNet architecture is adopted to jointly learn location related and communication features for beam prediction. Two fusion strategies, namely adaptive fusion and adversarial fusion, are introduced to enable efficient feature integration. The proposed framework is evaluated on datasets generated by the DeepMIMO simulator across four urban scenarios at 3.5 GHz following 3GPP specifications, where both reference signal received power and user equipment location information are available. Simulation results under both in-distribution and out-of-distribution settings demonstrate that the proposed approach consistently outperforms traditional baselines and achieves more accurate and robust beam prediction by effectively incorporating positioning information.

### Collaborative Spectrum Sensing in Cognitive and Intelligent Wireless Networks: An Artificial Intelligence Perspective

**Authors:** Peng Yi, Ying-Chang Liang
[arXiv](http://arxiv.org/abs/2602.09615v1) · [PDF](https://arxiv.org/pdf/2602.09615v1)

Artificial intelligence (AI) has become a key enabler for next-generation wireless communication systems, offering powerful tools to cope with the increasing complexity, dynamics, and heterogeneity of modern wireless environments. To illustrate the role and impact of AI in wireless communications, this paper takes collaborative spectrum sensing (CSS) in cognitive and intelligent wireless networks as a representative application and surveys recent advances from an AI perspective. We first introduce the fundamentals of CSS, including the general framework, classical detector design, and fusion strategies. Then, we present an overview of the state-of-the-art research on AI-driven CSS, classified into three categories: discriminative deep learning (DL) models, generative DL models, and deep reinforcement learning (DRL). Furthermore, we explore semantic communication (SemCom) as a promising solution for CSS, in which task-oriented representations are exchanged to reduce reporting overhead while preserving decision-critical information. Finally, we discuss limitations, open challenges, and future research directions at the intersection of AI and wireless communication.

### AI-Driven Cardiorespiratory Signal Processing: Separation, Clustering, and Anomaly Detection

**Authors:** Yasaman Torabi
[arXiv](http://arxiv.org/abs/2602.09210v1) · [PDF](https://arxiv.org/pdf/2602.09210v1)

This research applies artificial intelligence (AI) to separate, cluster, and analyze cardiorespiratory sounds. We recorded a new dataset (HLS-CMDS) and developed several AI models, including generative AI methods based on large language models (LLMs) for guided separation, explainable AI (XAI) techniques to interpret latent representations, variational autoencoders (VAEs) for waveform separation, a chemistry-inspired non-negative matrix factorization (NMF) algorithm for clustering, and a quantum convolutional neural network (QCNN) designed to detect abnormal physiological patterns. The performance of these AI models depends on the quality of the recorded signals. Therefore, this thesis also reviews the biosensing technologies used to capture biomedical data. It summarizes developments in microelectromechanical systems (MEMS) acoustic sensors and quantum biosensors, such as quantum dots and nitrogen-vacancy centers. It further outlines the transition from electronic integrated circuits (EICs) to photonic integrated circuits (PICs) and early progress toward integrated quantum photonics (IQP) for chip-based biosensing. Together, these studies show how AI and next-generation sensors can support more intelligent diagnostic systems for future healthcare.

### DNS: Data-driven Nonlinear Smoother for Complex Model-free Process

**Authors:** Fredrik Cumlin, Anubhab Ghosh, Saikat Chatterjee
[arXiv](http://arxiv.org/abs/2602.08560v1) · [PDF](https://arxiv.org/pdf/2602.08560v1)

We propose data-driven nonlinear smoother (DNS) to estimate a hidden state sequence of a complex dynamical process from a noisy, linear measurement sequence. The dynamical process is model-free, that is, we do not have any knowledge of the nonlinear dynamics of the complex process. There is no state-transition model (STM) of the process available. The proposed DNS uses a recurrent architecture that helps to provide a closed-form posterior of the hidden state sequence given the measurement sequence. DNS learns in an unsupervised manner, meaning the training dataset consists of only measurement data and no state data. We demonstrate DNS using simulations for smoothing of several stochastic dynamical processes, including a benchmark Lorenz system. Experimental results show that the DNS is significantly better than a deep Kalman smoother (DKS) and an iterative data-driven nonlinear state estimation (iDANSE) smoother.

### Foundation Model-Aided Hierarchical Deep Reinforcement Learning for Blockage-Aware Link in RIS-Assisted Networks

**Authors:** Mohammad Ghassemi, Han Zhang, Ali Afana, Akram Bin Sediq, Melike Erol-Kantarci
[arXiv](http://arxiv.org/abs/2602.09157v1) · [PDF](https://arxiv.org/pdf/2602.09157v1)

Reconfigurable intelligent surface (RIS) technology has the potential to significantly enhance the spectral efficiency (SE) of 6G wireless networks. However, practical deployment remains constrained by challenges in accurate channel estimation and control optimization under dynamic conditions. This paper presents a foundation model-aided hierarchical deep reinforcement learning (FM-HDRL) framework designed for joint beamforming and phase-shift optimization in RIS-assisted wireless networks. To implement this, we first fine-tune a pre-trained large wireless model (LWM) to translate raw channel data into low-dimensional, context-aware channel state information (CSI) embeddings. Next, these embeddings are combined with user location information and blockage status to select the optimal communication path. The resulting features are then fed into an HDRL model, assumed to be implemented at a centralized controller, which jointly optimizes the base station (BS) beamforming vectors and the RIS phase-shift configurations to maximize SE. Simulation results demonstrate that the proposed FM-HDRL framework consistently outperforms baseline methods in terms of convergence speed, spectral efficiency, and scalability. According to the simulation results, our proposed method improves 7.82% SE compared to the FM-aided deep reinforcement learning (FM-DRL) approach and a substantial enhancement of about 48.66% relative to the beam sweeping approach.

### LocDreamer: World Model-Based Learning for Joint Indoor Tracking and Anchor Scheduling

**Authors:** Geng Wang, Zhouyou Gu, Shenghong Li, Peng Cheng, Jihong Park, Branka Vucetic, Yonghui Li
[arXiv](http://arxiv.org/abs/2602.08204v1) · [PDF](https://arxiv.org/pdf/2602.08204v1)

Accurate, resource-efficient localization and tracking enables numerous location-aware services in next-generation wireless networks. However, existing machine learning-based methods often require large labeled datasets while overlooking spectrum and energy efficiencies. To fill this gap, we propose LocDreamer, a world model (WM)-based framework for joint target tracking and scheduling of localization anchors. LocDreamer learns a WM that captures the latent representation of the target motion and localization environment, thereby generating synthetic measurements to imagine arbitrary anchor deployments. These measurements enable imagination-driven training of both the tracking model and the reinforcement learning (RL)-based anchor scheduler that activates only the most informative anchors, which significantly reduce energy and signaling costs while preserving high tracking accuracy. Experiments on a real-world indoor dataset demonstrate that LocDreamer substantially improves data efficiency and generalization, outperforming conventional Bayesian filter with random scheduling by 37% in tracking accuracy, and achieving 86% of the accuracy of same model trained directly on real data.

---

## astro-ph.IM

*Papers in astro-ph.IM*

### astromorph: Self-supervised machine learning pipeline for astronomical morphology analysis

**Authors:** Per Bjerkeli, Jouni Kainulainen, Maria Carmen Toribio, Leon Boschman, Otoniel Maya Lucas
[arXiv](http://arxiv.org/abs/2602.09223v1) · [PDF](https://arxiv.org/pdf/2602.09223v1)

Modern telescopes generate increasingly large and diverse datasets, often consisting of complex and morphologically rich structures. To efficiently explore such data requires automated methods that can extract and organize physically meaningful information, ideally without the need for extensive manual interaction. We aim to provide a user-friendly implementation of a self-supervised machine learning framework to explore morphological properties of large datasets, based on the BYOL (Bootstrap Your Own Latents) method. By enabling the generation of meaningful image embeddings without manually labelled data, the framework will enable key tasks such as clustering, anomaly detection, and similarity based exploration. In contrast to existing BYOL implementations, astromorph accommodates data of varying dimensions and resolutions, including both single-channel FITS images and multi-channel spectral cubes. The package is built with usability in mind, offering streamlined pipeline scripts for ease of use as well as deeper customization options via PyTorch-based classes. To demonstrate the utility of astromorph, we apply it in two contrasting science cases representing different astronomical domains: images of protoplanetary disks observed with ALMA, and infrared dark clouds observed with Spitzer and Herschel. In both cases, we demonstrate how astromorph produces scientifically meaningful embeddings that capture morphological differences and similarities across large samples. astromorph enables users to apply a robust, label-free approach for uncovering morphological patterns in astronomical datasets. The successful application to two markedly different datasets suggest that the pipeline is broadly applicable across a wide range of imaging-rich astronomical context, providing a user friendly tool for advancing discovery in observational astronomy.

---

## math-ph

*Papers in math-ph*

### Macroscopic approximation of tight-binding models near spectral degeneracies and validity for wave packet propagation

**Authors:** Guillaume Bal, Paul Cazeaux, Daniel Massatt, Solomon Quinn
[arXiv](http://arxiv.org/abs/2602.08073v1) · [PDF](https://arxiv.org/pdf/2602.08073v1)

This paper concerns the derivation and validity of macroscopic descriptions of wave packets supported in the vicinity of degenerate points $(K,E)$ in the dispersion relation of tight-binding models accounting for macroscopic variations. We show that such wave packets are well approximated over long times by macroscopic models with varying orders of accuracy. Our main applications are in the analysis of single- and multilayer graphene tight-binding Hamiltonians modeling macroscopic variations such as those generated by shear or twist. Numerical simulations illustrate the theoretical findings.

---

## physics.geo-ph

*Papers in physics.geo-ph*

### Wavelet Packet-Based Diffusion Model for Ground Motion Generation with Multi-Conditional Energy and Spectral Matching

**Authors:** Yi Ding, Su Chen, Jinjun Hu, Xiaohu Hu, Qingxu Zhao, Xiaojun Li
[arXiv](http://arxiv.org/abs/2602.07405v1) · [PDF](https://arxiv.org/pdf/2602.07405v1)

Temporal energy distribution strongly affects nonlinear structural response and cumulative damage. We propose a multi-conditional diffusion framework for ground motion synthesis that simultaneously matches temporal energy evolution and target response spectra. Wavelet packet decomposition provides the signal representation and enables direct waveform reconstruction via orthogonal filter banks. A Transformer-based conditional encoder with cross-attention integrates heterogeneous conditions, including spectral ordinates, Arias intensity, temporal parameters, and Husid curves. The framework adopts the Elucidating Diffusion Model (EDM) with second-order Heun sampling to improve inference efficiency without sacrificing quality. Tests on the NGA-West2 database show that explicit temporal-energy constraints markedly improve control of energy onset and significant duration while preserving spectrum matching and maintaining stable diversity sampling. The framework yields spectrum-compatible motions with realistic energy evolution and supports uncertainty quantification via conditional diversity sampling.

---

## quant-ph

*Papers in quant-ph*

### Consensus Protocols for Entanglement-Aware Scheduling in Distributed Quantum Neural Networks

**Authors:** Kuan-Cheng Chen, Samuel Yen-Chi Chen, Mahdi Chehimi, Felix Burt, Kin K. Leung
[arXiv](http://arxiv.org/abs/2602.06847v1) · [PDF](https://arxiv.org/pdf/2602.06847v1)

The realization of distributed quantum neural networks (DQNNs) over quantum internet infrastructures faces fundamental challenges arising from the fragile nature of entanglement and the demanding synchronization requirements of distributed learning. We introduce a Consensus-Entanglement-Aware Scheduling (CEAS) framework that co-designs quantum consensus protocols with adaptive entanglement management to enable robust synchronous training across distributed quantum processors. CEAS integrates fidelity-weighted aggregation, in which parameter updates are weighted by quantum Fisher information to suppress noisy contributions, with decoherence-aware entanglement scheduling that treats Bell pairs as perishable resources subject to exponential decay. The framework incorporates quantum-authenticated Byzantine fault tolerance, ensuring security against malicious nodes while maintaining compatibility with noisy intermediate-scale quantum (NISQ) constraints. Our theoretical analysis establishes convergence guarantees under heterogeneous noise conditions, while numerical simulations demonstrate that CEAS maintains 10-15 percentage points higher accuracy compared to entanglement-oblivious baselines under coordinated Byzantine attacks, achieving 90 percent Bell-pair utilization despite coherence time limitations. This work provides a foundational architecture for scalable distributed quantum machine learning, bridging quantum networking, distributed optimization, and early fault-tolerant quantum computation.

---

## cs.CV

*Papers in cs.CV*

### Reliable Mislabel Detection for Video Capsule Endoscopy Data

**Authors:** Julia Werner, Julius Oexle, Oliver Bause, Maxime Le Floch, Franz Brinkmann, Hannah Tolle, Jochen Hampe, Oliver Bringmann
[arXiv](http://arxiv.org/abs/2602.06938v1) · [PDF](https://arxiv.org/pdf/2602.06938v1)

The classification performance of deep neural networks relies strongly on access to large, accurately annotated datasets. In medical imaging, however, obtaining such datasets is particularly challenging since annotations must be provided by specialized physicians, which severely limits the pool of annotators. Furthermore, class boundaries can often be ambiguous or difficult to define which further complicates machine learning-based classification. In this paper, we want to address this problem and introduce a framework for mislabel detection in medical datasets. This is validated on the two largest, publicly available datasets for Video Capsule Endoscopy, an important imaging procedure for examining the gastrointestinal tract based on a video stream of lowresolution images. In addition, potentially mislabeled samples identified by our pipeline were reviewed and re-annotated by three experienced gastroenterologists. Our results show that the proposed framework successfully detects incorrectly labeled data and results in an improved anomaly detection performance after cleaning the datasets compared to current baselines.

### Multi-AD: Cross-Domain Unsupervised Anomaly Detection for Medical and Industrial Applications

**Authors:** Wahyu Rahmaniar, Kenji Suzuki
[arXiv](http://arxiv.org/abs/2602.05426v1) · [PDF](https://arxiv.org/pdf/2602.05426v1)

Traditional deep learning models often lack annotated data, especially in cross-domain applications such as anomaly detection, which is critical for early disease diagnosis in medicine and defect detection in industry. To address this challenge, we propose Multi-AD, a convolutional neural network (CNN) model for robust unsupervised anomaly detection across medical and industrial images. Our approach employs the squeeze-and-excitation (SE) block to enhance feature extraction via channel-wise attention, enabling the model to focus on the most relevant features and detect subtle anomalies. Knowledge distillation (KD) transfers informative features from the teacher to the student model, enabling effective learning of the differences between normal and anomalous data. Then, the discriminator network further enhances the model's capacity to distinguish between normal and anomalous data. At the inference stage, by integrating multi-scale features, the student model can detect anomalies of varying sizes. The teacher-student (T-S) architecture ensures consistent representation of high-dimensional features while adapting them to enhance anomaly detection. Multi-AD was evaluated on several medical datasets, including brain MRI, liver CT, and retina OCT, as well as industrial datasets, such as MVTec AD, demonstrating strong generalization across multiple domains. Experimental results demonstrated that our approach consistently outperformed state-of-the-art models, achieving the best average AUROC for both image-level (81.4% for medical and 99.6% for industrial) and pixel-level (97.0% for medical and 98.4% for industrial) tasks, making it effective for real-world applications.

### PatchFlow: Leveraging a Flow-Based Model with Patch Features

**Authors:** Boxiang Zhang, Baijian Yang, Xiaoming Wang, Corey Vian
[arXiv](http://arxiv.org/abs/2602.05238v1) · [PDF](https://arxiv.org/pdf/2602.05238v1)

Die casting plays a crucial role across various industries due to its ability to craft intricate shapes with high precision and smooth surfaces. However, surface defects remain a major issue that impedes die casting quality control. Recently, computer vision techniques have been explored to automate and improve defect detection. In this work, we combine local neighbor-aware patch features with a normalizing flow model and bridge the gap between the generic pretrained feature extractor and industrial product images by introducing an adapter module to increase the efficiency and accuracy of automated anomaly detection. Compared to state-of-the-art methods, our approach reduces the error rate by 20\% on the MVTec AD dataset, achieving an image-level AUROC of 99.28\%. Our approach has also enhanced performance on the VisA dataset , achieving an image-level AUROC of 96.48\%. Compared to the state-of-the-art models, this represents a 28.2\% reduction in error. Additionally, experiments on a proprietary die casting dataset yield an accuracy of 95.77\% for anomaly detection, without requiring any anomalous samples for training. Our method illustrates the potential of leveraging computer vision and deep learning techniques to advance inspection capabilities for the die casting industry

---

## eess.IV

*Papers in eess.IV*

### AS-Mamba: Asymmetric Self-Guided Mamba Decoupled Iterative Network for Metal Artifact Reduction

**Authors:** Bowen Ning, Zekun Zhou, Xinyi Zhong, Zhongzhen Wang, HongXin Wu, HaiTao Wang, Liu Shi, Qiegen Liu
[arXiv](http://arxiv.org/abs/2602.06350v1) · [PDF](https://arxiv.org/pdf/2602.06350v1)

Metal artifact significantly degrades Computed Tomography (CT) image quality, impeding accurate clinical diagnosis. However, existing deep learning approaches, such as CNN and Transformer, often fail to explicitly capture the directional geometric features of artifacts, leading to compromised structural restoration. To address these limitations, we propose the Asymmetric Self-Guided Mamba (AS-Mamba) for metal artifact reduction. Specifically, the linear propagation of metal-induced streak artifacts aligns well with the sequential modeling capability of State Space Models (SSMs). Consequently, the Mamba architecture is leveraged to explicitly capture and suppress these directional artifacts. Simultaneously, a frequency domain correction mechanism is incorporated to rectify the global amplitude spectrum, thereby mitigating intensity inhomogeneity caused by beam hardening. Furthermore, to bridge the distribution gap across diverse clinical scenarios, we introduce a self-guided contrastive regularization strategy. Extensive experiments on public andclinical dental CBCT datasets demonstrate that AS-Mamba achieves superior performance in suppressing directional streaks and preserving structural details, validating the effectiveness of integrating physical geometric priors into deep network design.

---

## physics.chem-ph

*Papers in physics.chem-ph*

### End-to-End Differentiable Learning of a Single Functional for DFT and Linear-Response TDDFT

**Authors:** Xiaoyu Zhang
[arXiv](http://arxiv.org/abs/2602.05345v1) · [PDF](https://arxiv.org/pdf/2602.05345v1)

Density functional theory (DFT) and linear-response time-dependent density functional theory (LR-TDDFT) rely on an exchange-correlation (xc) approximation that provides not only energy but also its functional derivatives that enter the self-consistent potential and the response kernel. Here we present an end-to-end differentiable workflow to optimize a single deep-learned energy functional using targets from both Kohn-Sham DFT and adiabatic LR-TDDFT within the Tamm-Dancoff approximation. Implemented in a JAX-based two-component quantum chemistry code (IQC), the learned functional yields a consistent potential and LR kernel via automatic differentiation, enabling gradient-based training through the SCF fixed point and the Casida equation. As a proof of concept in a fixed finite basis (cc-pVDZ), we learn an exchange-correlation functional on the helium spectrum while incorporating one-electron self-interaction cancelation and the Lieb-Oxford inequality as penalty terms, and we assess its possible transfer to molecular test cases.

---
