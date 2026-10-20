
Research Plan: Anti-Shortcut Intervention for Mental-Health Models
核心问题我会定成：
When mental-health models exploit dataset-specific shortcuts, where should we intervene—on the data, training objective, representation, or inference process—to recover construct-robust generalization?
这比“我们提出一个 debiasing algorithm”更适合你现在的阶段，因为你还不知道哪类方法真正适合你观察到的 shortcut。
RQ
RQ
问题
RQ1
不同 anti-shortcut interventions 能否真正降低 shortcut dependence？
RQ2
Data-side intervention 和 model-side intervention 哪个更有效？
RQ3
减少 shortcut 后，模型有没有保留真正的 mental-health signal？
RQ4
intervention 是否改善 cross-dataset / OOD generalization，而不是只改善原 dataset？
RQ5
哪些 intervention 对不同 shortcut/model 都稳定？

真正关键的是 RQ3。
因为：
[
\text{shortcut reliance}\downarrow
]
并不一定意味着：
[
\text{construct learning}\uparrow.
]
模型可能只是什么都不会了。

Phase 0 — Freeze the evaluation before trying methods
既然前三部分已经做了，这一步不要继续“发现 shortcut”，而是把现有结果 freeze 成统一 evaluation suite。
假设：
[
X=(C,S)
]
其中：
(C)：真正应该预测 depression/mental-health construct 的信息
(S)：shortcut
(Y)：target label
至少固定四个 test conditions：
Test
目的
IID test
保证 normal performance 没坏
Shortcut-conflict test
(S) 与 (Y) correlation 被打破
Counterfactual pairs
固定 (C)，只改变 (S)
Cross-dataset test
shortcut distribution 自然发生变化

然后所有 mitigation 只能使用完全相同的 test sets。
这会让后面的 method comparison 非常干净。

Phase 1 — Data-side interventions
我建议至少跑三个，而不是只跑 adversarial augmentation。
D0. ERM baseline
普通 fine-tuning：
[
L=L_{\text{CE}}(Y,\hat Y)
]
它是所有 intervention 的 denominator。
D1. Shortcut removal / masking
最naive baseline。
例如 shortcut 是 sentiment / interviewer information / lexical cue：
[
X=(C,S)
\rightarrow
X'=(C,\varnothing)
]
它不一定是好方法，但论文里非常重要：
是不是单纯把 shortcut 删掉就够了？

D2. Rebalancing
重新采样，使：
[
P(S|Y=1)\approx P(S|Y=0).
]
这也是一个很强的 classical baseline。
ACL 2024 的 concept-level work就是使用生成的 counterfactual data 去 rebalance concept-label distribution，并取得比简单 token removal 更好的 robustness。(ACL Anthology)

D3. Counterfactual / counterbias augmentation ★
这是我最建议作为 data-side main method 的。
构造：
[
(C,S,Y)
]
和
[
(C,S',Y).
]
只改变 shortcut：
[
S\rightarrow S'
]
但：
[
C,Y=\text{fixed}.
]
例如：
symptom evidence 相同
sentiment / style / interviewer cue 改变
或者：
acoustic depression signal 相同
semantic positivity 改变。
DepFlow 做的 CDoA 本质上正是这种 controlled mismatch：把 depressive acoustic pattern 和 positive/neutral semantic content 配对，从而制造自然数据中较少出现的 acoustic-semantic conflict。(arXiv)
而 EMNLP 2025 CoBA 则把 counterfactual augmentation推广为 semantic-triple-level 的 counterbias augmentation。(ACL Anthology)

Phase 2 — Model-side: objective-level methods
这里我会先跑你已经读过的三个。
M1. PoE ★
这是非常值得先跑的。
训练：
Step 1
训练 weak/bias model：
[
f_b(S)\rightarrow Y.
]
甚至可以故意限制它只能看到 shortcut。
比如：
[
S=\text{sentiment/style/interviewer cue}.
]
Step 2
main model：
[
f_m(X)\rightarrow Y.
]
训练时：
[
p_{\text{PoE}}(y|x)
\propto
p_m(y|x)p_b(y|S).
]
然后 optimize PoE prediction。
直觉是：
如果一个 example 靠 shortcut 就能答对：
[
p_b(y|S)\approx1,
]
main model 获得的 learning pressure 较弱。
但如果：
[
S\rightarrow \text{wrong answer},
]
weak model confident but wrong，main model会获得很大的 gradient，必须找到别的 signal 来抵消它。ICLR 2021 对这个 gradient mechanism 有明确分析。
在 LLM 上怎么做？
不需要真正造 MoE architecture。
对于 decoder LLM，只取两个 label 的 logits：
[
z_m(\text{depressed}),\quad z_m(\text{control})
]
weak model也产生：
[
z_b(\text{depressed}),\quad z_b(\text{control})
]
然后：
[
z_{\text{final}}=z_m+\alpha z_b.
]
对：
[
CE(\text{softmax}(z_{\text{final}}),Y)
]
训练 main LLM，可以只训练 LoRA。
所以 PoE 完全可以搬到 LLM classification setting。

M2. Group DRO ★
Group DRO 最适合这种情况：
你已经知道 shortcut attribute (S)。
定义 group：
[
g=(Y,S).
]
例如：
Y
shortcut
depressed
positive
depressed
negative
control
positive
control
negative

ERM：
[
\min_\theta E[L].
]
Group DRO：
[
\min_\theta \max_g E[L|g].
]
也就是说：
不允许模型通过把 minority/conflict group 做得特别烂来获得高 overall accuracy。
原始 Group DRO work 发现，单纯使用 group DRO 对 overparameterized network 并不够，regularization/early stopping 对 worst-group generalization 很重要。(arXiv)
所以你的 implementation 应该是：
ERM
vs
Group DRO
vs
Group DRO + strong regularization / early stopping.
这一项很重要。

M3. Adversarial disentanglement ★
这里基本直接借鉴 EMNLP 2025 depression work。
Encoder：
[
h=f_\theta(X).
]
两个 heads：
[
h\rightarrow Y
]
预测 depression；
另一个：
[
h\rightarrow S
]
预测 shortcut。
加入 Gradient Reversal Layer：
[
L =
L_{\text{mental health}}
\lambda L_{\text{shortcut}}.
]
于是 encoder 希望：
[
I(h;Y)\uparrow
]
同时：
[
I(h;S)\downarrow.
]
EMNLP 2025 的 depression paper正是让 representation 对 interviewer-question type 保持 invariant。(ACL Anthology)
这其实比 DepFlow 更适合直接作为我们的 model-side baseline。

Phase 3 — Representation / architecture interventions
前三个跑完之后，再进入比较新的方法。
M4. Causal Attention Tuning
EMNLP 2025 CAT 的想法是：
不要只告诉模型：
“这个 shortcut 不好。”
而是直接控制 attention：
[
A_{\text{causal}}\uparrow,
\qquad
A_{\text{spurious}}\downarrow.
]
他们在 Llama-3.1-8B、Qwen 等模型上做了 causal attention tuning，并报告明显的 OOD improvement。(ACL Anthology)
对于我们：
symptom-relevant span
vs
shortcut span
如果已经可以标出来，CAT 就非常合理。
优点：
intervention 很直接。
缺点：
需要可靠的 token/span-level causal annotation。
所以我把它放第二梯队。

M5. Representation debiasing / PRIFT ★
这是 ACL 2026 的方法，也是我觉得特别值得我们尝试的 model-side intervention。
假设 embedding：
[
h=h_C+h_S.
]
找到 shortcut/confounder direction：
[
v_S.
]
然后：
[
h'
h-\operatorname{Proj}_{v_S}(h).
]
目标：
[
h'\approx h_C.
]
PRIFT 用 natural-language prompts 构建 confounder prototypes，再在 latent geometry 里删除 instance-specific confounding deviations；ACL 2026 paper 在多个 text classification benchmark 上验证，并在 CivilComments 上报告相对于 ERM 超过 20-point 的 worst-group accuracy improvement。(ACL Anthology)
它特别适合：
dataset identity
sentiment
interviewer style
demographic cue
source/platform cue
这种可以形成 latent direction 的 shortcut。

Phase 4 — Prompt/inference intervention
M6. CoT / anti-shortcut prompting
这应该跑，但我不会把它当主方法。
至少比较：
Zero-shot
CoT
Explicit anti-shortcut instruction
例如：
Base your prediction only on clinically relevant evidence and do not infer depression from sentiment, writing style, or interviewer wording alone.
EMNLP 2024 Shortcut Suite 发现 CoT 对多类 shortcut dependence 有明显缓解作用。(ACL Anthology)
它的价值主要是：
如果一个 prompt 就能解决问题，我们还有没有必要 retrain？
这是很好的 low-cost baseline。

Phase 5 — 你问的“直接控制 optimization”到底是什么？
这个其实比听起来简单很多。
它不是新的 architecture。
它问的是：
能不能仅仅改变模型“怎么训练”，就让 shortcut 更难被学到？
ICLR 2026 的 Implicit Regularization of SGD Reduces Shortcut Learning 发现：
[
\text{larger learning rate}
+
\text{smaller batch}
]
会增强 SGD implicit regularization，并减少 spurious-feature reliance。(OpenReview)
所以你的 M7 可以极其简单：
LR
Batch
small
large
small
small
large
large
large
small

再测：
[
\text{shortcut sensitivity}.
]
也可以加入：
weight decay
early stopping
这类 intervention。
但是我不会把它作为 main method。
理由是它目前的理论/实验结果不是专门针对 mental-health LLM fine-tuning；对我们来说更像一个非常便宜的 diagnostic baseline：
shortcut robustness 是否部分只是 optimization dynamics 的结果？
如果有效，再深入。

所以第一轮真正应该跑的是这个 matrix
ID
Intervention
Side
Cost
Priority
B0
ERM
baseline
★
必须
D1
shortcut removal
Data
★
必须
D2
rebalancing
Data
★
必须
D3
counterfactual augmentation
Data
★★
最高
M1
PoE
Objective
★★
最高
M2
Group DRO
Objective
★★
最高
M3
adversarial disentanglement
Representation
★★
最高
M4
CAT
Attention
★★★
第二轮
M5
PRIFT
Representation
★★★
较高
M6
CoT / prompting
Inference
★
必须
M7
LR/batch/early stopping
Optimization
★
必须但只做小 sweep

我不建议第一天就把 10 个方法 × 所有模型 × 所有 datasets 全跑起来。

Phase 6 — Screening experiment
第一轮只选：
[
\boxed{\text{1 canonical shortcut}}
]
和：
[
\boxed{\text{1 encoder model + 1 decoder LLM}}
]
这是非常重要的。
比如 shortcut (S_1) 选你们目前证据最强、最好 manipulate 的一个。
然后：
[
10\ interventions
\times
2\ models
\times
3\ seeds.
]
全部统一 training budget。
先得到一张：
Method
IID
Conflict
CF sensitivity ↓
Worst-group
Cross-dataset
ERM










Removal










Rebalance










CDA










PoE










Group DRO










Adversarial










CAT










PRIFT










Optimization












不要只看 F1：定义一个 intervention success criterion
我建议至少保留五个指标。
1. IID utility
[
\Delta F1_{\text{IID}}.
]
不能 robustness 提高 10%，IID 掉 20%。

2. Conflict performance
shortcut 与 label 冲突时：
[
F1_{\text{conflict}}.
]
这是最直接指标。

3. Counterfactual Shortcut Sensitivity
对于：
[
(x,x')
]
二者只有 shortcut 不一样：
[
CSS=
|p(Y|x)-p(Y|x')|.
]
越低越好。

4. Worst-group performance
[
\min_g F1_g.
]
特别适合 Group DRO 等方法。

5. Construct Preservation ★★★
这个应该成为你们 paper 的特色。
制造：
[
C\rightarrow C'
]
但：
[
S=\text{fixed}.
]
模型应该对真正 symptom evidence 的变化敏感。
所以理想 intervention：
[
\underbrace{\frac{\partial \hat Y}{\partial S}}_{\text{shortcut sensitivity}}
\downarrow
]
同时：
[
\underbrace{\frac{\partial \hat Y}{\partial C}}_{\text{construct sensitivity}}
\uparrow
\text{ or unchanged}.
]
这比只报 OOD F1 强得多。

Phase 7 — 只让 Top 2–3 晋级
第一轮之后，不要继续十个方法都做到底。
设置一个简单 rule：
方法必须同时满足
Conflict/OOD ↑
shortcut sensitivity ↓
construct sensitivity 不下降
IID performance 不显著下降
然后留下 Top 2–3。
我目前先验上最看好：
[
\boxed{\text{Counterfactual augmentation}}
]
vs
[
\boxed{\text{Group DRO / PoE}}
]
vs
[
\boxed{\text{Adversarial disentanglement / PRIFT}}
]
也就是说刚好代表：
data → objective → representation
三个 intervention level。

Phase 8 — 最重要的 experiment：组合方法
这里反而可能出现你最后的 new method。
假设结果发现：
Counterfactual augmentation 最强：
[
D3
]
Group DRO 第二：
[
M2.
]
那就试：
[
\boxed{D3+M2}
]
即：
construct-preserving counterfactual augmentation + robust objective
或者：
[
\boxed{D3+\text{Adversarial disentanglement}}.
]
这有很清楚的解释：
Data intervention：
[
P_{\text{train}}(S,Y)
]
被修复；
Model intervention：
[
I(h;S)
]
进一步受到约束。
于是你的最终贡献可能不是简单说：
“counterfactual augmentation works.”
而是发现：
Data- and model-side interventions address complementary failure modes: data intervention breaks shortcut correlations, while representation/objective intervention prevents residual shortcut encoding.
这个就开始像一篇完整方法论文了。

