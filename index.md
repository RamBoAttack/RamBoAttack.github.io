## RamBoAttack Against Deep Neural Network

Reproduce our results: [GitHub](https://github.com/RamBoAttack/RamBoAttack.github.io)

Check out our paper: [RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit](https://arxiv.org/abs/2112.05282)

Cite our research: 
```
@inproceedings{Vo2022,
    title = {RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit},
    year = {2022},
    journal = {Network and Distributed Systems Security (NDSS) Symposium},
    author = {Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe},
}
```

#### ABSTRACT

Machine  learning  models  are  critically  susceptibleto  evasion  attacks  from  adversarial  examples.  Generally,  ad-versarial  examples—modified  inputs  deceptively  similar  to  the original  input—are  constructed  under  whitebox  access  settings by  adversaries  with  full  access  to  the  model.  However,  recent attacks  have  shown  a  remarkable  reduction  in  the  number  ofqueries  to  craft  adversarial  examples  using  blackbox  attacks. Particularly  alarming  is  the  now, practical,  ability  to  exploitsimply the classification decision (hard label only) from a trainedmodel’saccess   interfaceprovided   by   a   growing   number   of Machine  Learning  as  a  Service  (MLaaS)  providers—including Google, Microsoft, IBM—and used by a plethora of applications in corporating  these  models.  An  adversary’s  ability  to  exploitonly the predicted label from a model-query to craft adversarial examples  is  distinguished  as  a decision-based attack.

In   our   study,   we   first   deep-dive   into   recent   state-of-the-art  decision-based  attacks  in  ICLR  and  S&P  to  highlight  the costly nature of discovering low distortion adversarial employing approximate  gradient  estimation  methods.  We  develop  a robust class  of query  efficient attacks  capable  of  avoiding  entrapment in a local minimum and misdirections from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack,  exploits  the  notion  of  Randomized  Block  Coordi-nate Descent to explore the hidden classifier manifold, targeting perturbations  to  manipulate  only  localized  input  features  to address  the  issues  of  gradient  estimation  methods.  Importantly, the RamBoAttack is  demonstrably  more  robust  to  the  different sample inputs available to an adversary and/or the targeted class. Overall,  for  a  given  target  class, RamBoAttackis  demonstrated to be more robust at achieving a lower distortion within a given query  budget.  We  curate  our  extensive  results  using  the  large-scale  high  resolution ImageNet dataset  and  open-source  our attack,  test  samples  and  artifacts  onGitHub

#### AN ILLUSTRATION OF RAMBOATTACK

![Figure 1](image/high level and hybrid-approach explain demo.svg#gh-dark-mode-only){:height="125%" width="125%"}

<!--
![Figure 1](image/high level and hybrid-approach explain demo.svg#gh-dark-mode-only){:height="700px" width="400px"}
![Figure 1](image/high level and hybrid-approach explain demo.svg#gh-dark-mode-only){:class="img-responsive"}
-->

Figure 1: A pictorial illustration of RamBoAttack to craft an adversarial example. In a targeted attack, the first component (GradEstimation) initializes an attack with a starting image from a target class (e.g. we use a clip art _street lamp_ for illustration) and then manipulates this image to search for adversarial examples that looks like an image from source class e.g _traffic light_. The attack switches to the second component, BlockDescent, when it reaches its own local minimum. BlockDescent helps to redirect away from that local minimum by manipulating multiple blocks---or making local changes to the current adversarial example. Subsequently, the adversarial example crafted by  BlockDescent is refined by the third component (GradEstimation).

#### VISUALIZATION

![Figure 2](image/gh-hard case visualization-Page-12.svg#gh-dark-mode-only)

Figure  2:  An  illustration  ofhardcase  (**white stork** to **goldfish**)  versusnon-hardcase  (**white stork** to **digital watch**)  on ImageNet. Adversarial  examples  in non-hard cases  and hard cases  are  yielded  after  50K  and  100K  queries,  respectively.  Except  for  Boundary  attack,  adversarial examples crafted by different attacks in non-hard cases are slightly different whilst in the hard case, our RamBoAttack is able to craft an adversarial example with much smaller distortion than other attacks due to the ability of our BlockDescent formulation to target effective localized perturbations.

![Figure 3](image/gh-hard case visualization-Page-14.svg#gh-dark-mode-only)

Figure 3: Grad-CAM tool visualizes salient features of the starting image or target class: **digital watch**. Perturbation heat map (PHM) visualizes the normalized perturbation magnitude at each pixel. Comparing different pertur-bations crafted by different attacks highlights that the localized perturbations yielded  by  RamBoAttack  concentrate  on  salient  areas  illustrated  by  GRAD-CAM  and  embeds  these  targeted  perturbations  in  the  source  image  to  fool the classifier to predict the target class; even though, RamBoAttack does not exploit the knowledge of salient regions to generate perturbations.

![Figure 4](image/crafting adv ex visualization.svg#gh-dark-mode-only)

Figure 4: An illustration of different distortion levels produced by RamBoAttack. The first row demonstrates an example from _CIFAR10_ with a starting image of a __dog__ gradually perturbed until it is similar to the source image __car__---the adversarial example. The bottom row demonstrates an example from _ImageNet_ with is a starting image of a __digital watch__ gradually perturbed until it is similar to the source image __white stork__---the adversarial example..
