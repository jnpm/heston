---
layout: page
title: data output
permalink: /output/
---

Most of the console output for the empirical analysis has been saved into csv files and are available in the [github repo](https://github.com/jnpm/heston/tree/main/docs/data){:target="_blank"}.

| Description | File |
| ------- | ----------- |
| Option pricing with Black-Scholes closed form| BSModelPrice |
| Absolute price difference between model prices calculated via numerical integration and market prices | HestonError |
| Implied volatilities extracted from model option prices calculated via Monte Carlo simulation | MCModelIV |
| Heston model option pricing via Monte Carlo simulation | MCPrice |
| Absolute percentage error of MC option prices and market prices | MCRealHestonError |
| Heston model option pricing via numerical integration | ModelPrice |
| Implied volatilities extracted from model option prices calculated via numerical integration | NIModelIV |
| MSE and MAPE of implied volatilities and option prices | Other Measures |
| Absolute percentage error of model option prices calculated via numerical integration and market prices | RelHestonError |
| Heston model parameters | param |