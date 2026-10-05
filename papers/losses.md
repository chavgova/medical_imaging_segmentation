
## `healthcare-09-00938.pdf`

**Sugino et al. (2021)**
*Loss Weightings for Improving Imbalanced Brain Structure Segmentation Using Fully Convolutional Networks*

This study uses the CE and Dice loss, but also does weighting to fix the imbalance of classes in the segmentation task. The weighted variants of CE and Dice they use are:

Baseline losses (their eq. 1-2), with $g_{i,c}$ the ground-truth label and $p_{i,c}$ the predicted probability of class $c$ at pixel $i$, $N$ pixels and $C$ classes:

$$L_{\text{CE}} = -\frac{1}{N} \sum_{c=1}^{C} \sum_{i=1}^{N} g_{i,c} \log p_{i,c}$$

$$L_{\text{Dice}} = 1 - \frac{2 \sum_{c=1}^{C} \sum_{i=1}^{N} g_{i,c} p_{i,c}}{\sum_{c=1}^{C} \sum_{i=1}^{N} g_{i,c} + \sum_{c=1}^{C} \sum_{i=1}^{N} p_{i,c}}$$

Inverse frequency weighting (their eq. 3, Table 1), weight per class $c$:

$$W_c^{\text{Inverse}} = \frac{1}{\left(\sum_{i=1}^{N} g_{i,c}\right)^{\alpha}}$$

with $\alpha = 1$ for CE and $\alpha = 2$ for Dice (the latter is what's known as the Generalized Dice loss). Applied to both losses:

$$L_{\text{CE}}^{\text{Inverse}} = -\frac{1}{N} \sum_{c=1}^{C} W_c^{\text{Inverse}} \sum_{i=1}^{N} g_{i,c} \log p_{i,c}$$

$$L_{\text{Dice}}^{\text{Inverse}} = 1 - \frac{2 \sum_{c=1}^{C} W_c^{\text{Inverse}} \sum_{i=1}^{N} g_{i,c} p_{i,c}}{\sum_{c=1}^{C} W_c^{\text{Inverse}} \sum_{i=1}^{N} (g_{i,c} + p_{i,c})}$$

They also give inverse *median* frequency weighting (their eq. 4-5), with $F_c$ the normalized frequency of class $c$:

$$F_c = \frac{\sum_{i=1}^{N} g_{i,c}}{N}, \qquad W_c^{\text{Median}} = \frac{\text{median}(F_c)}{F_c}$$

## `EMS126388-2.pdf`

**Sudre et al. (2017)**
*Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations*

This one is cited by Sugino et al. (2021) as the source for inverse frequency weighting. The weighted CE and Dice losses they give are:

Weighted Cross-Entropy (WCE), two-class form, with $r_n$ the reference/ground-truth label and $p_n$ the predicted foreground probability over $N$ elements:

$$\text{WCE} = -\frac{1}{N} \sum_{n=1}^{N} w \, r_n \log(p_n) + (1 - r_n) \log(1 - p_n)$$

where the foreground weight $w$ is defined from the predicted probabilities themselves (not the ground truth):

$$w = \frac{N - \sum_n p_n}{\sum_n p_n}$$

Generalized Dice Loss (GDL), for $L$ labels with per-label weight $w_l$:

$$\text{GDL} = 1 - 2 \frac{\sum_{l=1}^{L} w_l \sum_n r_{ln} p_{ln}}{\sum_{l=1}^{L} w_l \sum_n (r_{ln} + p_{ln})}$$

Weight used ($\text{GDL}_V$): $w_l = 1 / \left(\sum_n r_{ln}\right)^2$.
Same as Sugino et al.'s inverse frequency weighting with $\alpha = 2$.

## `09125-XuF.pdf`

**Xu et al. (2025)**
*A Unified Loss for Handling Inter-Class and Intra-Class Imbalance in Medical Image Segmentation*

This paper is more recent, and goes over intra-class imbalance as well. I.e., within a certain class some pixels are easy to guess (middle of an organ, middle of background), and some are harder to guess (on the edge between organ and background).

**Inter-CBL** (eq. 11) inter-class imbalance fix, replaces WCE. $F$: set of foreground pixels, $f = |F|$. $B$: set of background pixels. $H \subset B$: the $f$ hardest (highest-loss) background pixels, so $|H| = f$. $B \setminus H$: the remaining, easy background pixels.

$$\text{Inter-CBL} = -\frac{1}{f} \sum_{i \in F} \log(p_i) - \frac{1}{f} \sum_{j \in H} \log(1 - p_j) - \frac{1}{N} \sum_{k \in B \setminus H} \log(1 - p_k)$$

**Intra-CBL** (eq. 12) intra-class imbalance fix, replaces TopK. $t$: threshold splitting each class's own pixels into easy ($p_i^c > t$) and hard ($p_i^c \le t$) groups, both kept (unlike TopK, nothing is discarded).

$$\text{Intra-CBL} = -\sum_{c=1}^{C} \left( \frac{\sum_i \mathbb{I}\{y_i = c, p_i^c \le t\} \log(p_i^c)}{\sum_i \mathbb{I}\{y_i = c, p_i^c \le t\}} + \frac{\sum_i \mathbb{I}\{y_i = c, p_i^c > t\} \log(p_i^c)}{\sum_i \mathbb{I}\{y_i = c, p_i^c > t\}} \right)$$

**Balance loss** (eq. 13) the actual proposed loss, $\alpha = 0.5$ default:

$$\text{BL} = \alpha \cdot \text{Intra-CBL} + (1 - \alpha) \cdot \text{Inter-CBL}$$
