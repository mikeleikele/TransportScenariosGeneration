<h1 style="color:#1d66ab;">Experiment Cases (num_case)</h1>
---
layout: default
title: Available num_case Values
---

This page describes the different values of the <code>num_case</code> parameter that can be used to select the experimental case in the traffic scenario generation project.

---

## What is `num_case`?

`num_case` is a numeric parameter that allows you to choose among various predefined cases for running experiments. Each case corresponds to a specific configuration, such as the dataset used, the model, and other relevant settings.

---

## List of Available Studies Cases


## METR-LA DATASET
**Model: AE+GAN**

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Case</th>
      <th>Model Case</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_METR_16</td>
    </tr>
    <tr>
      <td>2</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_METR_32</td>
    </tr>
    <tr>
      <td>3</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_METR_48</td>
    </tr>
    <tr>
      <td>4</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_METR_64</td>
    </tr>
  </tbody>
</table>



## PEMS_BAY DATASET

**Model: AE+GAN**
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Case</th>
      <th>Model Case</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_PEMS_16</td>
    </tr>
    <tr>
      <td>6</td>
      <td>PEMS_METR</td>
      <td>AE&gt;WGAN_linear_pretrained_PEMS_16</td>
    </tr>
    <tr>
      <td>7</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_PEMS_32</td>
    </tr>
    <tr>
      <td>8</td>
      <td>PEMS_METR</td>
      <td>AE&gt;WGAN_linear_pretrained_PEMS_32</td>
    </tr>
    <tr>
      <td>9</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_PEMS_48</td>
    </tr>
    <tr>
      <td>10</td>
      <td>PEMS_METR</td>
      <td>AE&gt;GAN_linear_pretrained_PEMS_64</td>
    </tr>
  </tbody>
</table>


## CHENGDU DATASET

**Model: AE+GAN Fully Connected**

| ID  | Case      | Model Case                          |
|-------|-------------------|---------------------------------------|
| 11    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_16_A_linear |
| 12    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_32_A_linear |
| 13    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_48_A_linear |
| 14    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_64_A_linear |
| 15    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_96_A_linear |
| 16    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_128_A_linear|
| 17    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_192_A_linear|
| 18    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_256_A_linear|
| 19    | CHENGDU_SMALLGRAPH| AE>GAN_CHENGDU_SMALLGRAPH_512_A_linear|

**Model: AE+GAN with GraphConvolutional layers**
| ID  | Case      | Model Case                          |
|-------|-------------------|---------------------------------------|
| gvae_0016 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_16_A_graph  |
| gvae_0032 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_32_A_graph  |
| gvae_0048 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_48_A_graph  |
| gvae_0064 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_64_A_graph  |
| gvae_0096 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_96_A_graph  |
| gvae_0128 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_128_A_graph |
| gvae_0192 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_192_A_graph |
| gvae_0256 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_256_A_graph |
| gvae_0512 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_512_A_graph |
| gvae_1024 | CHENGDU_SMALLGRAPH | AE>GAN_CHENGDU_SMALLGRAPH_1024_A_graph|

**Model: VAE with GraphConvolutional layers**
| ID  | Case      | Model Case                          |
|-------|-------------------|---------------------------------------|
| vae_0016 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_fix  |
| vae_0032 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_fix  |
| vae_0048 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_fix  |
| vae_0064 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_64_A_graph_kl_fix  |
| vae_0096 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_96_A_graph_kl_fix  |
| vae_0128 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_128_A_graph_kl_fix |
| vae_0192 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_192_A_graph_kl_fix |
| vae_0256 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_256_A_graph_kl_fix |
| vae_0512 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_512_A_graph_kl_fix |
| vae_1024 | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_1024_A_graph_kl_fix|

**Model: B-VAE with GraphConvolutional layers**
B-VAE with B funciotion: linear from 0 to 1

| ID  | Case      | Model Case                          |
|-------|-------------------|---------------------------------------|
| bvae_0016_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_lin   |
| bvae_0032_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_lin   |
| bvae_0048_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_lin   |
| bvae_0064_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_64_A_graph_kl_lin   |
| bvae_0096_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_96_A_graph_kl_lin   |
| bvae_0128_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_128_A_graph_kl_lin  |
| bvae_0192_lin | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_192_A_graph_kl_lin  |

B-VAE with B funciotion: cos with period 20 epochs

| ID  | Case      | Model Case                          |
|-------|-------------------|---------------------------------------|
| bvae_0016_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_cos  |
| bvae_0032_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_cos  |
| bvae_0048_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_cos  |
| bvae_0064_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_64_A_graph_kl_cos  |
| bvae_0096_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_96_A_graph_kl_cos  |
| bvae_0128_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_128_A_graph_kl_cos |
| bvae_0192_cos | CHENGDU_SMALLGRAPH | VAE_CHENGDU_SMALLGRAPH_192_A_graph_kl_cos |


## How to Use

To run an experiment with a specific case, use the `--num_case` parameter with the desired value. For example:

```bash
python3 test.py --num_case 1 --experiment_name_suffix METR16_exp --main_folder experiments --repeat 5 --optimization yes --load_model no --train_models yes