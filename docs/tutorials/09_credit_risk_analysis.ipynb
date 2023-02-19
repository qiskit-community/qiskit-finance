{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Risk Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Introduction\n",
    "This tutorial shows how quantum algorithms can be used for credit risk analysis.\n",
    "More precisely, how Quantum Amplitude Estimation (QAE) can be used to estimate risk measures with a quadratic speed-up over classical Monte Carlo simulation.\n",
    "The tutorial is based on the following papers:\n",
    "\n",
    "- [Quantum Risk Analysis. Stefan Woerner, Daniel J. Egger.](https://www.nature.com/articles/s41534-019-0130-6) [Woerner2019]\n",
    "- [Credit Risk Analysis using Quantum Computers. Egger et al. (2019)](https://arxiv.org/abs/1907.03044) [Egger2019]\n",
    "\n",
    "A general introduction to QAE can be found in the following paper:\n",
    "\n",
    "- [Quantum Amplitude Amplification and Estimation. Gilles Brassard et al.](http://arxiv.org/abs/quant-ph/0005055)\n",
    "\n",
    "The structure of the tutorial is as follows:\n",
    "\n",
    "1. [Problem Definition](#Problem-Definition)\n",
    "2. [Uncertainty Model](#Uncertainty-Model)\n",
    "3. [Expected Loss](#Expected-Loss)\n",
    "4. [Cumulative Distribution Function](#Cumulative-Distribution-Function)\n",
    "5. [Value at Risk](#Value-at-Risk)\n",
    "6. [Conditional Value at Risk](#Conditional-Value-at-Risk)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from qiskit import QuantumRegister, QuantumCircuit\n",
    "from qiskit.circuit.library import IntegerComparator\n",
    "from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem\n",
    "from qiskit_aer.primitives import Sampler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Problem Definition\n",
    "\n",
    "In this tutorial we want to analyze the credit risk of a portfolio of $K$ assets.\n",
    "The default probability of every asset $k$ follows a *Gaussian Conditional Independence* model, i.e., given a value $z$ sampled from a latent random variable $Z$ following a standard normal distribution, the default probability of asset $k$ is given by\n",
    "\n",
    "$$p_k(z) = F\\left( \\frac{F^{-1}(p_k^0) - \\sqrt{\\rho_k}z}{\\sqrt{1 - \\rho_k}} \\right) $$\n",
    "\n",
    "where $F$ denotes the cumulative distribution function of $Z$, $p_k^0$ is the default probability of asset $k$ for $z=0$ and $\\rho_k$ is the sensitivity of the default probability of asset $k$ with respect to $Z$. Thus, given a concrete realization of $Z$ the individual default events are assumed to be independent from each other.\n",
    "\n",
    "We are interested in analyzing risk measures of the total loss\n",
    "\n",
    "$$ L = \\sum_{k=1}^K \\lambda_k X_k(Z) $$\n",
    "\n",
    "where $\\lambda_k$ denotes the _loss given default_ of asset $k$, and given $Z$, $X_k(Z)$ denotes a Bernoulli variable representing the default event of asset $k$. More precisely, we are interested in the expected value $\\mathbb{E}[L]$, the Value at Risk (VaR) of $L$ and the Conditional Value at Risk of $L$ (also called Expected Shortfall). Where VaR and CVaR are defined as\n",
    "\n",
    "$$ \\text{VaR}_{\\alpha}(L) = \\inf \\{ x \\mid \\mathbb{P}[L <= x] \\geq 1 - \\alpha \\}$$\n",
    "\n",
    "with confidence level $\\alpha \\in [0, 1]$, and\n",
    "\n",
    "$$ \\text{CVaR}_{\\alpha}(L) = \\mathbb{E}[ L \\mid L \\geq \\text{VaR}_{\\alpha}(L) ].$$\n",
    "\n",
    "For more details on the considered model, see, e.g.,<br>\n",
    "[Regulatory Capital Modeling for Credit Risk. Marek Rutkowski, Silvio Tarca](https://arxiv.org/abs/1412.1183)\n",
    "\n",
    "\n",
    "\n",
    "The problem is defined by the following parameters:\n",
    "\n",
    "- number of qubits used to represent $Z$, denoted by $n_z$\n",
    "- truncation value for $Z$, denoted by $z_{\\text{max}}$, i.e., Z is assumed to take $2^{n_z}$ equidistant values in $\\{-z_{max}, ..., +z_{max}\\}$ \n",
    "- the base default probabilities for each asset $p_0^k \\in (0, 1)$, $k=1, ..., K$\n",
    "- sensitivities of the default probabilities with respect to $Z$, denoted by $\\rho_k \\in [0, 1)$\n",
    "- loss given default for asset $k$, denoted by $\\lambda_k$\n",
    "- confidence level for VaR / CVaR $\\alpha \\in [0, 1]$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# set problem parameters\n",
    "n_z = 2\n",
    "z_max = 2\n",
    "z_values = np.linspace(-z_max, z_max, 2**n_z)\n",
    "p_zeros = [0.15, 0.25]\n",
    "rhos = [0.1, 0.05]\n",
    "lgd = [1, 2]\n",
    "K = len(p_zeros)\n",
    "alpha = 0.05"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Uncertainty Model\n",
    "\n",
    "We now construct a circuit that loads the uncertainty model. This can be achieved by creating a quantum state in a register of $n_z$ qubits that represents $Z$ following a standard normal distribution. This state is then used to control single qubit Y-rotations on a second qubit register of $K$ qubits, where a $|1\\rangle$ state of qubit $k$ represents the default event of asset $k$. The resulting quantum state can be written as\n",
    "\n",
    "$$ |\\Psi\\rangle = \\sum_{i=0}^{2^{n_z}-1} \\sqrt{p_z^i} |z_i \\rangle \\bigotimes_{k=1}^K \n",
    "\\left( \\sqrt{1 - p_k(z_i)}|0\\rangle + \\sqrt{p_k(z_i)}|1\\rangle\\right),\n",
    "$$\n",
    "\n",
    "where we denote by $z_i$ the $i$-th value of the discretized and truncated $Z$ [Egger2019]."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel as GCI\n",
    "\n",
    "u = GCI(n_z, z_max, p_zeros, rhos)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace\">     ┌───────┐\n",
       "q_0: ┤0      ├\n",
       "     │       │\n",
       "q_1: ┤1      ├\n",
       "     │  P(X) │\n",
       "q_2: ┤2      ├\n",
       "     │       │\n",
       "q_3: ┤3      ├\n",
       "     └───────┘</pre>"
      ],
      "text/plain": [
       "     ┌───────┐\n",
       "q_0: ┤0      ├\n",
       "     │       │\n",
       "q_1: ┤1      ├\n",
       "     │  P(X) │\n",
       "q_2: ┤2      ├\n",
       "     │       │\n",
       "q_3: ┤3      ├\n",
       "     └───────┘"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "u.draw()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now use the simulator to validate the circuit that constructs $|\\Psi\\rangle$ and compute the corresponding exact values for\n",
    "\n",
    "- expected loss $\\mathbb{E}[L]$\n",
    "- PDF and CDF of $L$ \n",
    "- value at risk $VaR(L)$ and corresponding probability\n",
    "- conditional value at risk $CVaR(L)$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "u_measure = u.measure_all(inplace=False)\n",
    "sampler = Sampler()\n",
    "job = sampler.run(u_measure)\n",
    "binary_probabilities = job.result().quasi_dists[0].binary_probabilities()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# analyze uncertainty circuit and determine exact solutions\n",
    "p_z = np.zeros(2**n_z)\n",
    "p_default = np.zeros(K)\n",
    "values = []\n",
    "probabilities = []\n",
    "num_qubits = u.num_qubits\n",
    "\n",
    "for i, prob in binary_probabilities.items():\n",
    "    # extract value of Z and corresponding probability\n",
    "    i_normal = int(i[-n_z:], 2)\n",
    "    p_z[i_normal] += prob\n",
    "\n",
    "    # determine overall default probability for k\n",
    "    loss = 0\n",
    "    for k in range(K):\n",
    "        if i[K - k - 1] == \"1\":\n",
    "            p_default[k] += prob\n",
    "            loss += lgd[k]\n",
    "\n",
    "    values += [loss]\n",
    "    probabilities += [prob]\n",
    "\n",
    "\n",
    "values = np.array(values)\n",
    "probabilities = np.array(probabilities)\n",
    "\n",
    "expected_loss = np.dot(values, probabilities)\n",
    "losses = np.sort(np.unique(values))\n",
    "pdf = np.zeros(len(losses))\n",
    "for i, v in enumerate(losses):\n",
    "    pdf[i] += sum(probabilities[values == v])\n",
    "cdf = np.cumsum(pdf)\n",
    "\n",
    "i_var = np.argmax(cdf >= 1 - alpha)\n",
    "exact_var = losses[i_var]\n",
    "exact_cvar = np.dot(pdf[(i_var + 1) :], losses[(i_var + 1) :]) / sum(pdf[(i_var + 1) :])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Expected Loss E[L]:                0.6396\n",
      "Value at Risk VaR[L]:              2.0000\n",
      "P[L <= VaR[L]]:                    0.9570\n",
      "Conditional Value at Risk CVaR[L]: 3.0000\n"
     ]
    }
   ],
   "source": [
    "print(\"Expected Loss E[L]:                %.4f\" % expected_loss)\n",
    "print(\"Value at Risk VaR[L]:              %.4f\" % exact_var)\n",
    "print(\"P[L <= VaR[L]]:                    %.4f\" % cdf[exact_var])\n",
    "print(\"Conditional Value at Risk CVaR[L]: %.4f\" % exact_cvar)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "tags": [
     "nbsphinx-thumbnail"
    ]
   },
   "outputs": [
    {
     "data": {
      "image/png": "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iz3f2dkZHTp0QEREBOLi4lCjRo0C+69fv445c+agTp06CAkJQXZ2Nu7cuYNBgwZh586dyMnJwYIFC+Do6Fio7IiICABA79699aoTEelJICIyMgAlLrNmzdIcHxsbK4wePVrw9vYWbG1tBXd3d2HAgAHCyZMnC5W9fft2ISwsTGjYsKHg6uoqODg4CAEBAcKoUaOEa9euFTj2f//7nzB06FChbt26grOzs+Ds7Cw0aNBAmDhxohAXF6dzfUaMGFFifXx9fXUqa/369QIA4auvviq078mTJ8KSJUuEHj16CLVq1RKkUqkAQHBychLatWsnbNiwochyQ0JCBFtbW+Hhw4c614uI9CcShHzPxomIyKQUCgV8fX1RpUoVzYCYRYmIiMDs2bM1T5OKEhcXB19fXwwePBgbN240YrRE9CL2cSIiKkNSqRQzZ87ExYsXsXPnTqOUuXDhQojFYnz++edGKY+IisbEiYiojI0ZMwYBAQGYM2dOqcuKj4/HypUrMXr06CKnciEi4+GrOiIiIiId8YkTERERkY6YOBERERHpiIkTERERkY6YOBERERHpiIkTERERkY6YOBERERHpiIkTERERkY6YOBERERHpiIkTERERkY6YOBERERHp6P9rha9UD688uQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot loss PDF, expected loss, var, and cvar\n",
    "plt.bar(losses, pdf)\n",
    "plt.axvline(expected_loss, color=\"green\", linestyle=\"--\", label=\"E[L]\")\n",
    "plt.axvline(exact_var, color=\"orange\", linestyle=\"--\", label=\"VaR(L)\")\n",
    "plt.axvline(exact_cvar, color=\"red\", linestyle=\"--\", label=\"CVaR(L)\")\n",
    "plt.legend(fontsize=15)\n",
    "plt.xlabel(\"Loss L ($)\", size=15)\n",
    "plt.ylabel(\"probability (%)\", size=15)\n",
    "plt.title(\"Loss Distribution\", size=20)\n",
    "plt.xticks(size=15)\n",
    "plt.yticks(size=15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot results for Z\n",
    "plt.plot(z_values, p_z, \"o-\", linewidth=3, markersize=8)\n",
    "plt.grid()\n",
    "plt.xlabel(\"Z value\", size=15)\n",
    "plt.ylabel(\"probability (%)\", size=15)\n",
    "plt.title(\"Z Distribution\", size=20)\n",
    "plt.xticks(size=15)\n",
    "plt.yticks(size=15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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k5MT/P39ERwcDDMzszodHxERETUvEkEQhMZuxMuioKAApqamyM/P55QKzVhpaSn27t2Ld999l3cEETVj7OsvD2V/f6vlmCoiIiKipkZtL/8REVFV7WbFNHYT6L90NQV81wPoHLofxWUcY6sObn7r16j755kqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEgFDFREREZEIGKqIiIiIRMBQRURERCQChioiIiIiETBUEREREYmAoYqIiIhIBAxVRERERCJgqCIiIiISAUMVERERkQgYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiYChioiIiEgEDFVEREREImCoIiIiIhIBQxURERGRCBiqiIiIiETAUEVEREQkAoYqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEgFDFREREZEIGKqIiIiIRMBQRURERCQChioiIiIiETBUEREREYmAoYqIiIhIBAxVRERERCJgqCIiIiISAUMVERERkQgYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiYChioiIiEgEDFVEREREImCoIiIiIhIBQxURERGRCBiqiIiIiETAUEVEREQkAoYqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEoFah6onT55g3rx5cHJygp6eHqysrBAUFITbt28rXcfDhw+xadMmjBo1Cvb29tDR0YGxsTF69uyJsLAwlJaWKty2rKwMK1asQJcuXaCvrw8LCwuMGDECly5dEuPwiIiIqBlR21D19OlT+Pj4YOHChXj06BECAgJgY2ODtWvXws3NDdevX1eqnqVLl2LMmDHYunUrzMzMMGTIEPTo0QPnz5/H9OnT4ePjg6KioirblZeXY/jw4QgJCUFmZib8/PzQqVMnREVFwd3dHYmJiWIfMhERETVhahuqFi1ahPj4eHh4eODq1avYunUrEhISsGzZMuTk5CAoKEipegwNDTFz5kzcvHkTycnJ2LJlC+Li4nDhwgXY2trir7/+wqJFi6psFxERgejoaDg6OuLy5cuIiorCkSNHsG3bNhQVFWHMmDF49uyZ2IdNRERETZRahqqSkhKEh4cDAFauXAkjIyNZWUhICFxcXHD06FEkJSXVWNfs2bPxf//3f7C1ta203NHREd9++y0AYPPmzVW2W758OQDgu+++Q+vWrWXLhw4dioEDB+LatWvYuXNn7Q+OiIiImiW1DFUnTpxAfn4+2rdvDzc3tyrlw4YNAwDs3r27TvtxdXUFAGRlZVVafuPGDVy6dAn6+vrw8/Ort/0TERFR86GWoer8+fMAgK5du8otly5PSUmp036k47LatGkjd/+dO3eGtrZ2ve2fiIiImg+1DFUZGRkAAGtra7nl0uXp6el12k9YWBgAICAgoFH2T0RERM2HVmM3QJ5Hjx4BAAwMDOSWGxoaAgAKCwtV3sfPP/+MgwcPokWLFpg1a1a97L+4uBjFxcWy1wUFBQCA0tLSaqdyoKZN+tnyM6b6oKspNHYT6L90NYRK/1Ljq6+fu8rWq5ahqr4dP34cwcHBkEgkiIiIgJWVVb3sZ8mSJViwYEGV5QcOHFAY2Kj5+PPPPxu7CdQMfdejsVtAL1roXt7YTaD/2rt3b73UK2/qJXnUMlRJ7/ZTdBCPHz8GABgbG9e67tTUVAQEBKCkpAQ//PADBg8eXG/7nz17NkJCQmSvCwoKYGNjA19fX5iYmNS67dQ0lJaW4s8//0S/fv3kjskjqovOofsbuwn0X7oaAha6l2PuGQ0Ul0sauzkEIDW0f73UK73SVBO1DFXS6Q8yMzPllkuX29nZ1areGzduwNfXF3l5eQgNDcUnn3xSr/vX1dWFrq5uleXa2tr8ZfsS4OdM9aG4jL+81U1xuYSfi5qor5+5ytarlgPVpVMdJCcnyy2XLndxcVG6zuzsbPTr1w/Z2dkIDg7G/Pnza9x/amqq3OuoquyfiIiImje1DFWenp4wNTVFWloazp07V6U8KioKAODv769UfXl5eejfvz/S0tLwwQcfYMWKFdWub29vj9deew1PnjxBTExMnfdPREREzZ9ahiodHR1MmzYNADB16lTZGCbg+UznKSkp8Pb2Rrdu3WTLw8PD4ezsjNmzZ1eqq6ioCH5+frhw4QJGjBiBNWvWQCKp+TStdCzUzJkzce/ePdnyP/74A7t27UKHDh2qTMVARERELy+1HFMFAHPmzMHBgwdx8uRJODo6wsvLC+np6UhISICFhQUiIiIqrZ+bm4srV64gOzu70vKvv/4ap06dgqamJrS0tPDhhx/K3V9kZGSl10FBQdi7dy+io6Ph7OyMPn36IDc3F0ePHoW+vj42bNgALS21ffuIiIiogaltKtDT08Phw4exZMkSbNq0CTt27IC5uTkCAwOxcOFChRNzvigvLw8AUFZWhk2bNilc78VQpaGhgW3btiEsLAwRERHYs2cPDA0NMXToUCxYsAAdO3ZU+diIiIio+ZEIgsBZyxpIQUEBTE1NkZ+fzykVmrHS0lLs3bsX7777Lu/+I9G1m1V1nCc1Dl1NAd/1KMPMRE3e/acmbn5b9Xm9YlD297dajqkiIiIiamoYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiYChioiIiEgEDFVEREREImCoIiIiIhIBQxURERGRCBiqiIiIiETAUEVEREQkAoYqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEgFDFREREZEIGKqIiIiIRMBQRURERCQChioiIiIiETBUEREREYlAqy4bl5SU4NKlS8jJycHDhw/RokULWFhY4LXXXoOOjo5YbSQiIiJSe7UOVTk5OYiMjERMTAwSExNRXFxcZR1dXV306NEDAwYMwPjx42FhYSFKY4mIiIjUldKh6tq1a5g7dy6io6NRUlICAGjZsiW6desGc3NzmJiYID8/H3l5ebh8+TKOHTuGY8eOYc6cORgyZAj++c9/okOHDvV2IERERESNSalQNW3aNKxZswZlZWV4++23MXr0aPTu3Rv29vYKt7l+/ToOHz6MTZs24ffff8f27dvx8ccf48cffxSt8URERETqQqmB6hEREZg8eTIyMjLw559/4oMPPqg2UAGAg4MDPvzwQ8TFxSE9PR2TJk1CRESEKI0mIiIiUjdKnam6fv062rRpo/JO2rZti7CwMMyePVvlOoiIiIjUmVJnquoSqOqjHiIiIiJ1w3mqiIiIiEQgWqhKSUnBuHHj4O7ujh49eiAoKAiXLl0Sq3oiIiIitSZKqNq2bRu6deuGnTt3QlNTE0VFRVi3bh1cXV0RGxsrxi6IiIiI1JoooWrmzJno378/bt++jYSEBKSmpuLMmTMwNDTk4HQiIiJ6KSgVqtasWaOw7OnTp7IpE4yMjGTL3dzc4OPjw0uARERE9FJQKlRNmjQJPXv2xOnTp6uU6enpwdTUFEeOHKm0/PHjxzh79izv+CMiIqKXglKh6q+//kJpaSk8PDwwYcIE5ObmViqfMmUKli9fjr59+2LWrFn49NNP0alTJ9y8eRNTpkypl4YTERERqROlQpWHhweSkpLw448/Ijo6Gq+++ipWrVoFQRAAAIsWLcLSpUtx6dIlfPfddwgPD0d5eTnCw8Mxc+bMej0AIiIiInWg9EB1iUSCyZMn4+rVqxg2bBg+/fRTdOvWDSdPnoREIkFISAhu376N/Px85OfnIyMjg2epiIiI6KVR67v/zM3N8csvvyA+Ph46Ojrw8vJCYGAgcnJyAADGxsYwNjYWvaFERERE6kzlKRXc3d0RHx+PNWvWIDY2Fo6OjggLC0N5ebmY7SMiIiJqEmoVqu7evYtDhw5h+/btOH36NEpKShAUFIQrV67g/fffx4wZM/D666/j6NGj9dVeIiIiIrWkVKgqLi7GlClTYGtri379+mH48OF444030KFDB0RFRcHU1BQ//vgjkpKS0KJFC/j4+GD06NHIysqq7/YTERERqQWlQtUXX3yBn3/+GW+//TY2btyIffv2Yfny5dDQ0MDIkSNx5swZAICLiwuOHTuGf//73zh69CicnZ3x/fff1+sBEBEREakDpULVli1b0LVrV8TGxmLkyJHo378/goODsXv3bpSXl2Pr1q2V1h8zZgyuXLmCjz/+GHPnzq2XhhMRERGpE6VC1ePHj9G6desqy6WzpT958qRKmZGREZYuXYpz587VrYVERERETYBSoertt9/G/v37sXTpUty7dw+lpaW4ePEigoKCIJFI4O3trXBbZ2dn0RpLREREpK6UClUrV66Ek5MTZs6cCUtLS+jp6cHFxQV79+7FhAkTMHz48PpuJxEREZFa01JmJTs7O6SmpmL79u04f/488vLyYGtri3/84x9wcXGp7zYSERERqT2lQhUAaGhoYPjw4TwrRURERCSHyjOqExEREdH/KBWq9u3bJ8rO9u7dK0o9REREROpGqVDl5+cHDw8P7Nq1C2VlZbXawbNnzxAdHY2ePXvC399fpUYSERERqTulQlVkZCSysrIwePBgtGnTBpMnT8aWLVuQlpYmd/1r165h8+bNmDhxItq0aYNhw4bh7t27iIyMFLPtRERERGpDqYHq48aNw3vvvYdVq1bhp59+wi+//ILVq1cDeD6AvUWLFjA2NkZhYSEePnyI8vJyAIAgCHBycsK8efMwceJE6Orq1t+REBERETUipe/+09XVxWeffYbPPvsMx44dw549e3D8+HGkpKTg/v37uH//PgBAX18frq6u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geLyREZGIigoSJVdEBERETUpKoWqyMhI/PXXXzWud+LECaxbt06VXRARERE1KaLOU/WikpISaGpq1ucuiIiIiNRCvYUqQRCQnJwMCwuL+toFERERkdpQeqC6j49PpdexsbFVlkk9e/YMaWlpuHPnDu/2IyIiopeC0qHqyJEjsv9LJBLcuXMHd+7cUbi+trY2BgwYgKVLl9apgURERERNgdKh6saNGwCeX9ZzcHDAsGHD8P3338tdV0dHBy1btoS2trY4rSQiIiJSc0qHKjs7O9n/58+fDzc3t0rLiIiIiF5mKk3+OX/+fLHbQURERNSk1euUCkREREQvC6VClYaGBrS0tHD16lUAgKamptJfWloqnQwjIiIialKUSjy2traQSCSygec2NjaQSPhEbiIiIiIppULVzZs3q31NRERE9LLjmCoiIiIiETBUEREREYlAqct/GRkZddqJra1tnbYnIiIiUndKhap27dqpPDBdIpHg2bNnKm1LRERE1FQoFap69erFu/2IiIiIqqFUqKr4MGUiIiIiqooD1YmIiIhEwFBFREREJAKlLv8dO3YMANCjRw/o6enJXiurV69etW8ZERERUROiVKjq3bs3JBIJLl26BCcnJ9lrZZWVlancQCIiIqKmQKlQNW7cOEgkEpiamlZ6TURERETPKRWqIiMjq31NRERE9LLjQHUiIiIiESh1pqomd+/eRVZWFgDAysoKrVu3FqNaIiIioiZD5TNVgiDghx9+gJOTE6ysrODu7g53d3dYWVnB0dERYWFhKC8vF7OtRERERGpLpTNVxcXF8Pf3R1xcHARBgJmZGezs7AA8f/hyWloaQkJCsGfPHuzZswe6urqiNpqIiIhI3ah0pmrx4sU4ePAgOnXqhH379uH+/ftITk5GcnIycnNzERsbi86dO+PQoUNYvHix2G0mIiIiUjsqhaoNGzagRYsWOHz4MPr371+l3NfXF3FxcTA1NcX69evr3EgiIiIidadSqMrKykKfPn3wyiuvKFynZcuW8PHxQXZ2tsqNIyIiImoqVApVbdu2RUlJSY3rlZaWwsrKSpVdEBERETUpKoWqMWPGIC4uDunp6QrXSU9PR1xcHEaPHq1y44iIiIiaCpVC1Zw5c+Dj44NevXohIiICjx8/lpU9fvwYa9euhbe3N/r06YN58+aJ1lgiIiIidaXUlAoODg5V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Ro1CSUkJpkyZgmfPnsnKZs6ciZycHIwdOxatWrWSLb98+TLWr19faToEAMjJycHIkSNx69YtuLq6wtPTU9TjJiIioqZLLQeqA8CcOXNw8OBBnDx5Eo6OjvDy8kJ6ejoSEhJgYWGBiIiISuvn5ubiypUrcsc5/etf/0J8fDy2b98OZ2dnuLu74+LFi0hNTYWjoyOWL19eaf07d+5g3LhxCA4Ohru7OywsLJCVlYWkpCQUFhbC2toav//+OyQSDkwkIiKi59TyTBUA6Onp4fDhw5g7dy4MDAywY8cOpKenIzAwEMnJyXBwcFC6rpYtWyIxMRGffPIJSkpKEB0djfz8fHz66adITEyEubl5pfWdnJwwffp0vPrqq7hw4QK2bduGM2fOwNHREfPnz0dKSgqcnJzEPmQiIiJqwiSCIHAq2AZSUFAAU1NT5Ofniz5PVbtZMaLWR6rT1RTwXY8yzEzU5G3WauTmt36N3QRRsK+rD/Z19VNf/VzZ399qe6aKiIiIqClhqCIiIiISAUMVERERkQgYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiYChioiIiEgEDFVEREREImCoIiIiIhIBQxURERGRCBiqiIiIiETAUEVEREQkAoYqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEgFDFREREZEIGKqIiIiIRMBQRURERCQChioiIiIiETBUEREREYmAoYqIiIhIBAxVRERERCJgqCIiIiISAUMVERERkQgYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiYChioiIiEgEDFVEREREImCoIiIiIhIBQxURERGRCBiqiIiIiETAUEVEREQkAoYqIiIiIhEwVBERERGJgKGKiIiISAQMVUREREQiYKgiIiIiEgFDFREREZEIGKqIiIiIRMBQRURERCQChioiIiIiETBUEREREYmAoYqIiIhIBAxVRERERCJgqCIiIiISAUMVERERkQgYqoiIiIhEwFBFREREJAKGKiIiIiIRMFQRERERiUCtQ9WTJ08wb948ODk5QU9PD1ZWVggKCsLt27drXVdeXh6Cg4NhZ2cHXV1d2NnZYfr06Xj48KHCbcrKyrBixQp06dIF+vr6sLCwwIgRI3Dp0qU6HBURERE1R2obqp4+fQofHx8sXLgQjx49QkBAAGxsbLB2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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot results for default probabilities\n",
    "plt.bar(range(K), p_default)\n",
    "plt.xlabel(\"Asset\", size=15)\n",
    "plt.ylabel(\"probability (%)\", size=15)\n",
    "plt.title(\"Individual Default Probabilities\", size=20)\n",
    "plt.xticks(range(K), size=15)\n",
    "plt.yticks(size=15)\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Expected Loss\n",
    "\n",
    "To estimate the expected loss, we first apply a weighted sum operator to sum up individual losses to total loss:\n",
    "\n",
    "$$ \\mathcal{S}: |x_1, ..., x_K \\rangle_K |0\\rangle_{n_S} \\mapsto |x_1, ..., x_K \\rangle_K |\\lambda_1x_1 + ... + \\lambda_K x_K\\rangle_{n_S}. $$\n",
    "\n",
    "The required number of qubits to represent the result is given by\n",
    "\n",
    "$$ n_s = \\lfloor \\log_2( \\lambda_1 + ... + \\lambda_K ) \\rfloor + 1. $$\n",
    "\n",
    "Once we have the total loss distribution in a quantum register, we can use the techniques described in [Woerner2019] to map a total loss $L \\in \\{0, ..., 2^{n_s}-1\\}$ to the amplitude of an objective qubit by an operator\n",
    "\n",
    "$$ | L \\rangle_{n_s}|0\\rangle \\mapsto \n",
    "| L \\rangle_{n_s} \\left( \\sqrt{1 - L/(2^{n_s}-1)}|0\\rangle + \\sqrt{L/(2^{n_s}-1)}|1\\rangle \\right), $$\n",
    "\n",
    "which allows to run amplitude estimation to evaluate the expected loss."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# add Z qubits with weight/loss 0\n",
    "from qiskit.circuit.library import WeightedAdder\n",
    "\n",
    "agg = WeightedAdder(n_z + K, [0] * n_z + lgd)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from qiskit.circuit.library import LinearAmplitudeFunction\n",
    "\n",
    "# define linear objective function\n",
    "breakpoints = [0]\n",
    "slopes = [1]\n",
    "offsets = [0]\n",
    "f_min = 0\n",
    "f_max = sum(lgd)\n",
    "c_approx = 0.25\n",
    "\n",
    "objective = LinearAmplitudeFunction(\n",
    "    agg.num_sum_qubits,\n",
    "    slope=slopes,\n",
    "    offset=offsets,\n",
    "    # max value that can be reached by the qubit register (will not always be reached)\n",
    "    domain=(0, 2**agg.num_sum_qubits - 1),\n",
    "    image=(f_min, f_max),\n",
    "    rescaling_factor=c_approx,\n",
    "    breakpoints=breakpoints,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Create the state preparation circuit:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace\">           ┌───────┐┌────────┐      ┌───────────┐\n",
       "  state_0: ┤0      ├┤0       ├──────┤0          ├\n",
       "           │       ││        │      │           │\n",
       "  state_1: ┤1      ├┤1       ├──────┤1          ├\n",
       "           │  P(X) ││        │      │           │\n",
       "  state_2: ┤2      ├┤2       ├──────┤2          ├\n",
       "           │       ││        │      │           │\n",
       "  state_3: ┤3      ├┤3       ├──────┤3          ├\n",
       "           └───────┘│  adder │┌────┐│  adder_dg │\n",
       "objective: ─────────┤        ├┤2   ├┤           ├\n",
       "                    │        ││    ││           │\n",
       "    sum_0: ─────────┤4       ├┤0 F ├┤4          ├\n",
       "                    │        ││    ││           │\n",
       "    sum_1: ─────────┤5       ├┤1   ├┤5          ├\n",
       "                    │        │└────┘│           │\n",
       "    carry: ─────────┤6       ├──────┤6          ├\n",
       "                    └────────┘      └───────────┘</pre>"
      ],
      "text/plain": [
       "           ┌───────┐┌────────┐      ┌───────────┐\n",
       "  state_0: ┤0      ├┤0       ├──────┤0          ├\n",
       "           │       ││        │      │           │\n",
       "  state_1: ┤1      ├┤1       ├──────┤1          ├\n",
       "           │  P(X) ││        │      │           │\n",
       "  state_2: ┤2      ├┤2       ├──────┤2          ├\n",
       "           │       ││        │      │           │\n",
       "  state_3: ┤3      ├┤3       ├──────┤3          ├\n",
       "           └───────┘│  adder │┌────┐│  adder_dg │\n",
       "objective: ─────────┤        ├┤2   ├┤           ├\n",
       "                    │        ││    ││           │\n",
       "    sum_0: ─────────┤4       ├┤0 F ├┤4          ├\n",
       "                    │        ││    ││           │\n",
       "    sum_1: ─────────┤5       ├┤1   ├┤5          ├\n",
       "                    │        │└────┘│           │\n",
       "    carry: ─────────┤6       ├──────┤6          ├\n",
       "                    └────────┘      └───────────┘"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# define the registers for convenience and readability\n",
    "qr_state = QuantumRegister(u.num_qubits, \"state\")\n",
    "qr_sum = QuantumRegister(agg.num_sum_qubits, \"sum\")\n",
    "qr_carry = QuantumRegister(agg.num_carry_qubits, \"carry\")\n",
    "qr_obj = QuantumRegister(1, \"objective\")\n",
    "\n",
    "# define the circuit\n",
    "state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, name=\"A\")\n",
    "\n",
    "# load the random variable\n",
    "state_preparation.append(u.to_gate(), qr_state)\n",
    "\n",
    "# aggregate\n",
    "state_preparation.append(agg.to_gate(), qr_state[:] + qr_sum[:] + qr_carry[:])\n",
    "\n",
    "# linear objective function\n",
    "state_preparation.append(objective.to_gate(), qr_sum[:] + qr_obj[:])\n",
    "\n",
    "# uncompute aggregation\n",
    "state_preparation.append(agg.to_gate().inverse(), qr_state[:] + qr_sum[:] + qr_carry[:])\n",
    "\n",
    "# draw the circuit\n",
    "state_preparation.draw()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before we use QAE to estimate the expected loss, we validate the quantum circuit representing the objective function by just simulating it directly and analyzing the probability of the objective qubit being in the $|1\\rangle$ state, i.e., the value QAE will eventually approximate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "state_preparation_measure = state_preparation.measure_all(inplace=False)\n",
    "sampler = Sampler()\n",
    "job = sampler.run(state_preparation_measure)\n",
    "binary_probabilities = job.result().quasi_dists[0].binary_probabilities()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Exact Expected Loss:   0.6396\n",
      "Exact Operator Value:  0.3740\n",
      "Mapped Operator value: 0.5376\n"
     ]
    }
   ],
   "source": [
    "# evaluate the result\n",
    "value = 0\n",
    "for i, prob in binary_probabilities.items():\n",
    "    if prob > 1e-6 and i[-(len(qr_state) + 1) :][0] == \"1\":\n",
    "        value += prob\n",
    "\n",
    "print(\"Exact Expected Loss:   %.4f\" % expected_loss)\n",
    "print(\"Exact Operator Value:  %.4f\" % value)\n",
    "print(\"Mapped Operator value: %.4f\" % objective.post_processing(value))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we run QAE to estimate the expected loss with a quadratic speed-up over classical Monte Carlo simulation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Exact value:    \t0.6396\n",
      "Estimated value:\t0.6716\n",
      "Confidence interval: \t[0.6028, 0.7403]\n"
     ]
    }
   ],
   "source": [
    "# set target precision and confidence level\n",
    "epsilon = 0.01\n",
    "alpha = 0.05\n",
    "\n",
    "problem = EstimationProblem(\n",
    "    state_preparation=state_preparation,\n",
    "    objective_qubits=[len(qr_state)],\n",
    "    post_processing=objective.post_processing,\n",
    ")\n",
    "# construct amplitude estimation\n",
    "ae = IterativeAmplitudeEstimation(\n",
    "    epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={\"shots\": 100})\n",
    ")\n",
    "result = ae.estimate(problem)\n",
    "\n",
    "# print results\n",
    "conf_int = np.array(result.confidence_interval_processed)\n",
    "print(\"Exact value:    \\t%.4f\" % expected_loss)\n",
    "print(\"Estimated value:\\t%.4f\" % result.estimation_processed)\n",
    "print(\"Confidence interval: \\t[%.4f, %.4f]\" % tuple(conf_int))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cumulative Distribution Function\n",
    "\n",
    "Instead of the expected loss (which could also be estimated efficiently using classical techniques) we now estimate the cumulative distribution function (CDF) of the loss.\n",
    "Classically, this either involves evaluating all the possible combinations of defaulting assets, or many classical samples in a Monte Carlo simulation. Algorithms based on QAE have the potential to significantly speed up this analysis in the future.\n",
    "\n",
    "To estimate the CDF, i.e., the probability $\\mathbb{P}[L \\leq x]$, we again apply $\\mathcal{S}$ to compute the total loss, and then apply a comparator that for a given value $x$ acts as\n",
    "\n",
    "$$ \\mathcal{C}: |L\\rangle_n|0> \\mapsto \n",
    "\\begin{cases} \n",
    "|L\\rangle_n|1> & \\text{if}\\quad L \\leq x \\\\\n",
    "|L\\rangle_n|0> & \\text{if}\\quad L > x.\n",
    "\\end{cases} $$\n",
    "\n",
    "The resulting quantum state can be written as\n",
    "\n",
    "$$ \\sum_{L = 0}^{x} \\sqrt{p_{L}}|L\\rangle_{n_s}|1\\rangle + \n",
    "\\sum_{L = x+1}^{2^{n_s}-1} \\sqrt{p_{L}}|L\\rangle_{n_s}|0\\rangle, $$\n",
    "\n",
    "where we directly assume the summed up loss values and corresponding probabilities instead of presenting the details of the uncertainty model.\n",
    "\n",
    "The CDF($x$) equals the probability of measuring $|1\\rangle$ in the objective qubit and QAE can be directly used to estimate it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace\">         ┌──────┐\n",
       "state_0: ┤0     ├\n",
       "         │      │\n",
       "state_1: ┤1     ├\n",
       "         │  cmp │\n",
       "compare: ┤2     ├\n",
       "         │      │\n",
       "     a0: ┤3     ├\n",
       "         └──────┘</pre>"
      ],
      "text/plain": [
       "         ┌──────┐\n",
       "state_0: ┤0     ├\n",
       "         │      │\n",
       "state_1: ┤1     ├\n",
       "         │  cmp │\n",
       "compare: ┤2     ├\n",
       "         │      │\n",
       "     a0: ┤3     ├\n",
       "         └──────┘"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# set x value to estimate the CDF\n",
    "x_eval = 2\n",
    "\n",
    "comparator = IntegerComparator(agg.num_sum_qubits, x_eval + 1, geq=False)\n",
    "comparator.draw()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_cdf_circuit(x_eval):\n",
    "    # define the registers for convenience and readability\n",
    "    qr_state = QuantumRegister(u.num_qubits, \"state\")\n",
    "    qr_sum = QuantumRegister(agg.num_sum_qubits, \"sum\")\n",
    "    qr_carry = QuantumRegister(agg.num_carry_qubits, \"carry\")\n",
    "    qr_obj = QuantumRegister(1, \"objective\")\n",
    "    qr_compare = QuantumRegister(1, \"compare\")\n",
    "\n",
    "    # define the circuit\n",
    "    state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, name=\"A\")\n",
    "\n",
    "    # load the random variable\n",
    "    state_preparation.append(u, qr_state)\n",
    "\n",
    "    # aggregate\n",
    "    state_preparation.append(agg, qr_state[:] + qr_sum[:] + qr_carry[:])\n",
    "\n",
    "    # comparator objective function\n",
    "    comparator = IntegerComparator(agg.num_sum_qubits, x_eval + 1, geq=False)\n",
    "    state_preparation.append(comparator, qr_sum[:] + qr_obj[:] + qr_carry[:])\n",
    "\n",
    "    # uncompute aggregation\n",
    "    state_preparation.append(agg.inverse(), qr_state[:] + qr_sum[:] + qr_carry[:])\n",
    "\n",
    "    return state_preparation\n",
    "\n",
    "\n",
    "state_preparation = get_cdf_circuit(x_eval)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Again, we first use quantum simulation to validate the quantum circuit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace\">           ┌───────┐┌────────┐        ┌───────────┐\n",
       "  state_0: ┤0      ├┤0       ├────────┤0          ├\n",
       "           │       ││        │        │           │\n",
       "  state_1: ┤1      ├┤1       ├────────┤1          ├\n",
       "           │  P(X) ││        │        │           │\n",
       "  state_2: ┤2      ├┤2       ├────────┤2          ├\n",
       "           │       ││        │        │           │\n",
       "  state_3: ┤3      ├┤3       ├────────┤3          ├\n",
       "           └───────┘│  adder │┌──────┐│  adder_dg │\n",
       "objective: ─────────┤        ├┤2     ├┤           ├\n",
       "                    │        ││      ││           │\n",
       "    sum_0: ─────────┤4       ├┤0     ├┤4          ├\n",
       "                    │        ││  cmp ││           │\n",
       "    sum_1: ─────────┤5       ├┤1     ├┤5          ├\n",
       "                    │        ││      ││           │\n",
       "    carry: ─────────┤6       ├┤3     ├┤6          ├\n",
       "                    └────────┘└──────┘└───────────┘</pre>"
      ],
      "text/plain": [
       "           ┌───────┐┌────────┐        ┌───────────┐\n",
       "  state_0: ┤0      ├┤0       ├────────┤0          ├\n",
       "           │       ││        │        │           │\n",
       "  state_1: ┤1      ├┤1       ├────────┤1          ├\n",
       "           │  P(X) ││        │        │           │\n",
       "  state_2: ┤2      ├┤2       ├────────┤2          ├\n",
       "           │       ││        │        │           │\n",
       "  state_3: ┤3      ├┤3       ├────────┤3          ├\n",
       "           └───────┘│  adder │┌──────┐│  adder_dg │\n",
       "objective: ─────────┤        ├┤2     ├┤           ├\n",
       "                    │        ││      ││           │\n",
       "    sum_0: ─────────┤4       ├┤0     ├┤4          ├\n",
       "                    │        ││  cmp ││           │\n",
       "    sum_1: ─────────┤5       ├┤1     ├┤5          ├\n",
       "                    │        ││      ││           │\n",
       "    carry: ─────────┤6       ├┤3     ├┤6          ├\n",
       "                    └────────┘└──────┘└───────────┘"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "state_preparation.draw()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "state_preparation_measure = state_preparation.measure_all(inplace=False)\n",
    "sampler = Sampler()\n",
    "job = sampler.run(state_preparation_measure)\n",
    "binary_probabilities = job.result().quasi_dists[0].binary_probabilities()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Operator CDF(2) = 0.9492\n",
      "Exact    CDF(2) = 0.9570\n"
     ]
    }
   ],
   "source": [
    "# evaluate the result\n",
    "var_prob = 0\n",
    "for i, prob in binary_probabilities.items():\n",
    "    if prob > 1e-6 and i[-(len(qr_state) + 1) :][0] == \"1\":\n",
    "        var_prob += prob\n",
    "\n",
    "print(\"Operator CDF(%s)\" % x_eval + \" = %.4f\" % var_prob)\n",
    "print(\"Exact    CDF(%s)\" % x_eval + \" = %.4f\" % cdf[x_eval])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we run QAE to estimate the CDF for a given $x$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Exact value:    \t0.9570\n",
      "Estimated value:\t0.9596\n",
      "Confidence interval: \t[0.9587, 0.9605]\n"
     ]
    }
   ],
   "source": [
    "# set target precision and confidence level\n",
    "epsilon = 0.01\n",
    "alpha = 0.05\n",
    "\n",
    "problem = EstimationProblem(state_preparation=state_preparation, objective_qubits=[len(qr_state)])\n",
    "# construct amplitude estimation\n",
    "ae_cdf = IterativeAmplitudeEstimation(\n",
    "    epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={\"shots\": 100})\n",
    ")\n",
    "result_cdf = ae_cdf.estimate(problem)\n",
    "\n",
    "# print results\n",
    "conf_int = np.array(result_cdf.confidence_interval)\n",
    "print(\"Exact value:    \\t%.4f\" % cdf[x_eval])\n",
    "print(\"Estimated value:\\t%.4f\" % result_cdf.estimation)\n",
    "print(\"Confidence interval: \\t[%.4f, %.4f]\" % tuple(conf_int))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Value at Risk\n",
    "\n",
    "In the following we use a bisection search and QAE to efficiently evaluate the CDF to estimate the value at risk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_ae_for_cdf(x_eval, epsilon=0.01, alpha=0.05, simulator=\"aer_simulator\"):\n",
    "\n",
    "    # construct amplitude estimation\n",
    "    state_preparation = get_cdf_circuit(x_eval)\n",
    "    problem = EstimationProblem(\n",
    "        state_preparation=state_preparation, objective_qubits=[len(qr_state)]\n",
    "    )\n",
    "    ae_var = IterativeAmplitudeEstimation(\n",
    "        epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={\"shots\": 100})\n",
    "    )\n",
    "    result_var = ae_var.estimate(problem)\n",
    "\n",
    "    return result_var.estimation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def bisection_search(\n",
    "    objective, target_value, low_level, high_level, low_value=None, high_value=None\n",
    "):\n",
    "    \"\"\"\n",
    "    Determines the smallest level such that the objective value is still larger than the target\n",
    "    :param objective: objective function\n",
    "    :param target: target value\n",
    "    :param low_level: lowest level to be considered\n",
    "    :param high_level: highest level to be considered\n",
    "    :param low_value: value of lowest level (will be evaluated if set to None)\n",
    "    :param high_value: value of highest level (will be evaluated if set to None)\n",
    "    :return: dictionary with level, value, num_eval\n",
    "    \"\"\"\n",
    "\n",
    "    # check whether low and high values are given and evaluated them otherwise\n",
    "    print(\"--------------------------------------------------------------------\")\n",
    "    print(\"start bisection search for target value %.3f\" % target_value)\n",
    "    print(\"--------------------------------------------------------------------\")\n",
    "    num_eval = 0\n",
    "    if low_value is None:\n",
    "        low_value = objective(low_level)\n",
    "        num_eval += 1\n",
    "    if high_value is None:\n",
    "        high_value = objective(high_level)\n",
    "        num_eval += 1\n",
    "\n",
    "    # check if low_value already satisfies the condition\n",
    "    if low_value > target_value:\n",
    "        return {\n",
    "            \"level\": low_level,\n",
    "            \"value\": low_value,\n",
    "            \"num_eval\": num_eval,\n",
    "            \"comment\": \"returned low value\",\n",
    "        }\n",
    "    elif low_value == target_value:\n",
    "        return {\"level\": low_level, \"value\": low_value, \"num_eval\": num_eval, \"comment\": \"success\"}\n",
    "\n",
    "    # check if high_value is above target\n",
    "    if high_value < target_value:\n",
    "        return {\n",
    "            \"level\": high_level,\n",
    "            \"value\": high_value,\n",
    "            \"num_eval\": num_eval,\n",
    "            \"comment\": \"returned low value\",\n",
    "        }\n",
    "    elif high_value == target_value:\n",
    "        return {\n",
    "            \"level\": high_level,\n",
    "            \"value\": high_value,\n",
    "            \"num_eval\": num_eval,\n",
    "            \"comment\": \"success\",\n",
    "        }\n",
    "\n",
    "    # perform bisection search until\n",
    "    print(\"low_level    low_value    level    value    high_level    high_value\")\n",
    "    print(\"--------------------------------------------------------------------\")\n",
    "    while high_level - low_level > 1:\n",
    "\n",
    "        level = int(np.round((high_level + low_level) / 2.0))\n",
    "        num_eval += 1\n",
    "        value = objective(level)\n",
    "\n",
    "        print(\n",
    "            \"%2d           %.3f        %2d       %.3f    %2d            %.3f\"\n",
    "            % (low_level, low_value, level, value, high_level, high_value)\n",
    "        )\n",
    "\n",
    "        if value >= target_value:\n",
    "            high_level = level\n",
    "            high_value = value\n",
    "        else:\n",
    "            low_level = level\n",
    "            low_value = value\n",
    "\n",
    "    # return high value after bisection search\n",
    "    print(\"--------------------------------------------------------------------\")\n",
    "    print(\"finished bisection search\")\n",
    "    print(\"--------------------------------------------------------------------\")\n",
    "    return {\"level\": high_level, \"value\": high_value, \"num_eval\": num_eval, \"comment\": \"success\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------------------------------------------------------\n",
      "start bisection search for target value 0.950\n",
      "--------------------------------------------------------------------\n",
      "low_level    low_value    level    value    high_level    high_value\n",
      "--------------------------------------------------------------------\n",
      "-1           0.000         1       0.753     3            1.000\n",
      " 1           0.753         2       0.959     3            1.000\n",
      "--------------------------------------------------------------------\n",
      "finished bisection search\n",
      "--------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "# run bisection search to determine VaR\n",
    "objective = lambda x: run_ae_for_cdf(x)\n",
    "bisection_result = bisection_search(\n",
    "    objective, 1 - alpha, min(losses) - 1, max(losses), low_value=0, high_value=1\n",
    ")\n",
    "var = bisection_result[\"level\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Estimated Value at Risk:  2\n",
      "Exact Value at Risk:      2\n",
      "Estimated Probability:    0.959\n",
      "Exact Probability:        0.957\n"
     ]
    }
   ],
   "source": [
    "print(\"Estimated Value at Risk: %2d\" % var)\n",
    "print(\"Exact Value at Risk:     %2d\" % exact_var)\n",
    "print(\"Estimated Probability:    %.3f\" % bisection_result[\"value\"])\n",
    "print(\"Exact Probability:        %.3f\" % cdf[exact_var])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Conditional Value at Risk\n",
    "\n",
    "Last, we compute the CVaR, i.e. the expected value of the loss conditional to it being larger than or equal to the VaR.\n",
    "To do so, we evaluate a piecewise linear objective function $f(L)$, dependent on the total loss $L$, that is given by\n",
    "\n",
    "$$\n",
    "f(L) = \\begin{cases} \n",
    "0 & \\text{if}\\quad L \\leq VaR \\\\\n",
    "L & \\text{if}\\quad L > VaR.\n",
    "\\end{cases}\n",
    "$$\n",
    "\n",
    "To normalize, we have to divide the resulting expected value by the VaR-probability, i.e. $\\mathbb{P}[L \\leq VaR]$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace\">        ┌────┐\n",
       "q158_0: ┤0   ├\n",
       "        │    │\n",
       "q158_1: ┤1   ├\n",
       "        │    │\n",
       "  q159: ┤2 F ├\n",
       "        │    │\n",
       "  a4_0: ┤3   ├\n",
       "        │    │\n",
       "  a4_1: ┤4   ├\n",
       "        └────┘</pre>"
      ],
      "text/plain": [
       "        ┌────┐\n",
       "q158_0: ┤0   ├\n",
       "        │    │\n",
       "q158_1: ┤1   ├\n",
       "        │    │\n",
       "  q159: ┤2 F ├\n",
       "        │    │\n",
       "  a4_0: ┤3   ├\n",
       "        │    │\n",
       "  a4_1: ┤4   ├\n",
       "        └────┘"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# define linear objective\n",
    "breakpoints = [0, var]\n",
    "slopes = [0, 1]\n",
    "offsets = [0, 0]  # subtract VaR and add it later to the estimate\n",
    "f_min = 0\n",
    "f_max = 3 - var\n",
    "c_approx = 0.25\n",
    "\n",
    "cvar_objective = LinearAmplitudeFunction(\n",
    "    agg.num_sum_qubits,\n",
    "    slopes,\n",
    "    offsets,\n",
    "    domain=(0, 2**agg.num_sum_qubits - 1),\n",
    "    image=(f_min, f_max),\n",
    "    rescaling_factor=c_approx,\n",
    "    breakpoints=breakpoints,\n",
    ")\n",
    "\n",
    "cvar_objective.draw()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<qiskit.circuit.instructionset.InstructionSet at 0x28ec1adf0>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# define the registers for convenience and readability\n",
    "qr_state = QuantumRegister(u.num_qubits, \"state\")\n",
    "qr_sum = QuantumRegister(agg.num_sum_qubits, \"sum\")\n",
    "qr_carry = QuantumRegister(agg.num_carry_qubits, \"carry\")\n",
    "qr_obj = QuantumRegister(1, \"objective\")\n",
    "qr_work = QuantumRegister(cvar_objective.num_ancillas - len(qr_carry), \"work\")\n",
    "\n",
    "# define the circuit\n",
    "state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, qr_work, name=\"A\")\n",
    "\n",
    "# load the random variable\n",
    "state_preparation.append(u, qr_state)\n",
    "\n",
    "# aggregate\n",
    "state_preparation.append(agg, qr_state[:] + qr_sum[:] + qr_carry[:])\n",
    "\n",
    "# linear objective function\n",
    "state_preparation.append(cvar_objective, qr_sum[:] + qr_obj[:] + qr_carry[:] + qr_work[:])\n",
    "\n",
    "# uncompute aggregation\n",
    "state_preparation.append(agg.inverse(), qr_state[:] + qr_sum[:] + qr_carry[:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Again, we first use quantum simulation to validate the quantum circuit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "state_preparation_measure = state_preparation.measure_all(inplace=False)\n",
    "sampler = Sampler()\n",
    "job = sampler.run(state_preparation_measure)\n",
    "binary_probabilities = job.result().quasi_dists[0].binary_probabilities()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Estimated CVaR: 4.7144\n",
      "Exact CVaR:     3.0000\n"
     ]
    }
   ],
   "source": [
    "# evaluate the result\n",
    "value = 0\n",
    "for i, prob in binary_probabilities.items():\n",
    "    if prob > 1e-6 and i[-(len(qr_state) + 1)] == \"1\":\n",
    "        value += prob\n",
    "\n",
    "# normalize and add VaR to estimate\n",
    "value = cvar_objective.post_processing(value)\n",
    "d = 1.0 - bisection_result[\"value\"]\n",
    "v = value / d if d != 0 else 0\n",
    "normalized_value = v + var\n",
    "print(\"Estimated CVaR: %.4f\" % normalized_value)\n",
    "print(\"Exact CVaR:     %.4f\" % exact_cvar)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we run QAE to estimate the CVaR."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# set target precision and confidence level\n",
    "epsilon = 0.01\n",
    "alpha = 0.05\n",
    "\n",
    "problem = EstimationProblem(\n",
    "    state_preparation=state_preparation,\n",
    "    objective_qubits=[len(qr_state)],\n",
    "    post_processing=cvar_objective.post_processing,\n",
    ")\n",
    "# construct amplitude estimation\n",
    "ae_cvar = IterativeAmplitudeEstimation(\n",
    "    epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={\"shots\": 100})\n",
    ")\n",
    "result_cvar = ae_cvar.estimate(problem)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Exact CVaR:    \t3.0000\n",
      "Estimated CVaR:\t3.2832\n"
     ]
    }
   ],
   "source": [
    "# print results\n",
    "d = 1.0 - bisection_result[\"value\"]\n",
    "v = result_cvar.estimation_processed / d if d != 0 else 0\n",
    "print(\"Exact CVaR:    \\t%.4f\" % exact_cvar)\n",
    "print(\"Estimated CVaR:\\t%.4f\" % (v + var))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-08-22T01:56:12.651056Z",
     "start_time": "2019-08-22T01:56:12.640412Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.23.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.11.2</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.20.0</td></tr><tr><td><code>qiskit</code></td><td>0.41.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.9.10</td></tr><tr><td>Python compiler</td><td>Clang 13.1.6 (clang-1316.0.21.2.5)</td></tr><tr><td>Python build</td><td>main, Aug  9 2022 18:26:17</td></tr><tr><td>OS</td><td>Darwin</td></tr><tr><td>CPUs</td><td>10</td></tr><tr><td>Memory (Gb)</td><td>64.0</td></tr><tr><td colspan='2'>Thu Feb 16 15:49:06 2023 JST</td></tr></table>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2023.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import qiskit.tools.jupyter\n",
    "\n",
    "%qiskit_version_table\n",
    "%qiskit_copyright"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "celltoolbar": "Tags",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
