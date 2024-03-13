![extralearning logo](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/extralearning-full-logo.png)

> Integrated ML frameworks to simplify code in projects.


![GitHub release (latest by date)](https://img.shields.io/badge/release-v1.1.0-green)
![language (python)](https://img.shields.io/badge/language-python-blue)
![license](https://img.shields.io/badge/license-GPL--3.0-orange)

See the [examples website](https://github.com/extra-learning/extralearning-examples).

#### What is a extralearning?
extralearning is a robust and all-encompassing solution designed to bring together prominent Machine Learning frameworks. It not only consolidates these frameworks but also incorporates advanced functionality aimed at optimizing and simplifying code in the realm of Machine Learning projects. With extralearning, users can experience a seamless and efficient development process, making it an invaluable tool for those engaged in the field of Machine Learning.

___

## Key Features
* **Framework Consolidation:** Integrate leading Machine Learning frameworks seamlessly.
  
* **Code Streamlining Functionality:** Introduce advanced features to simplify and optimize code in Machine Learning projects.

* **Efficient Development:** Enhance the development process for a seamless and streamlined experience.

* **Fast validation:** Automated Model Training and Evaluation Pipeline.

## Get Started

```shell
pip install extralearning==1.1.0
```

See the [examples notebooks](https://github.com/extra-learning/extralearning-examples).

### Classification example

```shell
from extralearning.supervised import Classification

model = Classification(random_state = 42,
                       n_jobs = -1,
                       ignore_warnings = True)
                       
model.fit_train(X, y, CV = 2, CV_Stratified = False, CV_params = None, verbose = True)
```

### Regression example

```shell
from extralearning.supervised import Regression

model = Regression(n_jobs = -1,
                   ignore_warnings = True)
                       
model.fit_train(X, y, CV = 2, CV_params = None, verbose = True)
```

### Summary examples

#### summary(pandas: bool)
Generate a summary of the data stored in the object.

![summary](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/modelsummary.png)

#### fold_summary()
Calculate the mean summary of data grouped by 'Fold' and 'Model'.

![summary](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/foldsummary.png)

#### best(metric: str, pandas: bool)
Retrieve the best-performing model based on the specified metric.

![summary](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/best.png)

#### top/bottom(n: int, metric: str, pandas: bool)
Retrieve the best-performing model based on the specified metric.

![summary](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/top.png)

#### mean/median(metric: str, pandas: bool)
Calculate the mean/median of models grouped by the specified metric.

![summary](https://raw.githubusercontent.com/extra-learning/extralearning/master/images/mean.png)

## Donations

extralearning is a freely available, open-source library crafted during my limited free time. If you find value in the project and wish to contribute to its ongoing development, kindly consider making a small donation. Your support is genuinely appreciated!

[Donate with PayPal](https://www.paypal.me/ILIAMFTW)

## Contributors

[Liam Arguedas](https://github.com/liamarguedas)

## License

extralearning was created by [Liam Arguedas](https://github.com/liamarguedas)
and is licensed under the [GPL-3.0 license](/LICENSE).
