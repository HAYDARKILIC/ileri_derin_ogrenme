# İleri Derin Öğrenme

Haydar Kılıç — Yapay Zeka Mühendisliği

[`derin_ogrenme`](https://github.com/HAYDARKILIC/deep_learning) dersinin lisansüstü seviyedeki devamı.
Her defter, modern bir yöntemin matematiğini türetir ve ardından onu NumPy ile sıfırdan kodlar;
böylece derste geçen her iddia çalıştırabileceğiniz, bozup deneyebileceğiniz ve doğrulayabileceğiniz
bir şeye dönüşür.

**Ön koşul:** `deep_learning` dersi (geri yayılım, ESA'lar, transformer'lar, VAE/GAN/difüzyon,
pekiştirmeli öğrenme, çizge sinir ağları, normalleştirici akışlar).

---

## Defterler

| # | Dosya | Konular |
|---|------|--------|
| 01 | `Ders01_Ileri_Optimizasyon.ipynb` | Hessian ve kondisyon sayısı, momentum/Nesterov, Adam ve AdamW, ısınma ve kosinüs programları, gradyan kırpma, keskinlik ve SAM, K-FAC ön koşullandırma |
| 02 | `Ders02_Olcekleme_Yasalari_ve_Mimariler.ipynb` | Kuvvet yasası uydurma, hesap-optimal (Chinchilla) dağılım, metrik yanılsaması olarak "ortaya çıkış", transformer parametre/FLOP bütçesi, Mixture-of-Experts ve yük dengeleme, ViT'e karşı ESA tümevarımsal önyargısı |
| 03 | `Ders03_Verimli_Dikkat_ve_Uzun_Baglam.ipynb` | Çevrimiçi softmax ve FlashAttention, KV önbelleği, MHA/MQA/GQA, RoPE ve ALiBi, kayan pencere ve seyrek desenler, doğrusal dikkat ve yinelemeli biçimi |
| 04 | `Ders04_Oz_Denetimli_Ogrenme.ipynb` | InfoNCE ve $\log N$ sınırı, sıcaklık, hizalanma/tekdüzelik, SimCLR, temsil çökmesi, BYOL'un yordayıcısı ve gradyan durdurması, MAE, CLIP, sondalama protokolleri |
| 05 | `Ders05_Aktarimli_Ogrenme_ve_PEFT.ipynb` | Öznitelik aktarılabilirliği, felaket boyutunda unutma, SVD'den başlayarak LoRA, adaptörler/prefix/BitFit, optimizasyon belleği muhasebesi, blok bazlı niceleme ve NF4, bilgi damıtma |
| 06 | `Ders06_Belirsizlik_ve_Kalibrasyon.ipynb` | Rastlantısal ve bilgisel belirsizlik, Laplace yaklaşımı, MC dropout, derin topluluklar, ECE ve güvenilirlik diyagramları, sıcaklık ölçekleme, konformal tahmin, uygun skorlama kuralları |
| 07 | `Ders07_Difuzyon_ve_Skor_Tabanli_Modeller.ipynb` | Kapalı formlu ileri süreç ve SNR, skor kestirimi olarak gürültü giderme, Tweedie formülü, ters SDE ve olasılık-akış ODE'si, DDIM, sınıflandırıcısız yönlendirme, $\epsilon$/$x_0$/$v$ parametrizasyonları |
| 08 | `Ders08_Ileri_Pekistirmeli_Ogrenme_ve_RLHF.ipynb` | Ölümcül üçlü, aşırı tahmin yanlılığı ve çift Q-öğrenme, temel değerler ve GAE, TRPO/PPO kırpması, Bradley–Terry ödül modelleri, KL-düzenlileştirilmiş optimum, DPO, ödül aşırı-optimizasyonu |
| 09 | `Ders09_Geometrik_Derin_Ogrenme.ipynb` | Değişmezlik ve eşdeğerlik, artırıma karşı mimari, mesaj geçişi ve toplayıcı ifade gücü, 1-WL tavanı, aşırı düzleşme ve aşırı sıkışma, çizge transformer'ları |
| 10 | `Ders10_Yorumlanabilirlik.ipynb` | Gradyan doyması, bütünleşik gradyanlar ve bütünlük, tam Shapley değerleri, saliency akıl sağlığı testleri, kontrol görevli ve müdahaleli doğrusal sondalar, süperpozisyon, seyrek otokodlayıcılar |
| 11 | `Ders11_Dayaniklilik_ve_Dagilim_Kaymasi.ipynb` | Düşmanca örnekler neden var, FGSM'e karşı PGD, düşmanca eğitim ve dayanıklılık–doğruluk takası, rastgeleleştirilmiş yumuşatma sertifikaları, sahte korelasyonlar ve grup DRO, ortak değişken kayması ve OOD skorları |
| 12 | `Ders12_Verimli_Egitim_ve_Cikarim.ipynb` | bf16'ya karşı fp16 ve kayıp ölçekleme, gradyan kontrol noktaları, veri/tensör/boru hattı paralelliği, ZeRO, halka all-reduce, kod çözme çatı çizgisi, spekülatif kod çözme, yapılandırılmış seyreklik |

---

## Defterlerin yazım ilkeleri

- **Önce türetme, sonra kod.** Her bölüm önce matematiği ortaya koyar, ardından onu kodlar.
- **Yalnızca NumPy.** Saatler süren eğitim döngüsü ve GPU gereksinimi yok; her hücre bir dizüstü
  bilgisayarda saniyeler içinde çalışır. Amaç kıyaslama sayısı değil, mekanizmadır.
- **İddialar öne sürülmez, doğrulanır.** Sayısal olarak doğrulanabilen her sonuç doğrulanmıştır:
  çevrimiçi softmax tam dikkatle, bütünleşik gradyanlar bütünlük aksiyomuyla, difüzyon skoru sonlu
  farkla, DPO analitik RLHF optimumuyla karşılaştırılır.
- **Başarısızlık kipleri gösterilir.** Gradyan durdurma olmadan çökme, ölümcül üçlü altında ıraksama,
  dengeleme kaybı olmadan uzman çökmesi, 1-WL tavanı, enerji tabanlı OOD skorlarının ters çalışması.

---

## Depo Yapısı

```
ileri_derin_ogrenme/
├── README.md
├── requirements.txt
├── LICENSE
├── Ders01_Ileri_Optimizasyon.ipynb
├── Ders02_Olcekleme_Yasalari_ve_Mimariler.ipynb
├── Ders03_Verimli_Dikkat_ve_Uzun_Baglam.ipynb
├── Ders04_Oz_Denetimli_Ogrenme.ipynb
├── Ders05_Aktarimli_Ogrenme_ve_PEFT.ipynb
├── Ders06_Belirsizlik_ve_Kalibrasyon.ipynb
├── Ders07_Difuzyon_ve_Skor_Tabanli_Modeller.ipynb
├── Ders08_Ileri_Pekistirmeli_Ogrenme_ve_RLHF.ipynb
├── Ders09_Geometrik_Derin_Ogrenme.ipynb
├── Ders10_Yorumlanabilirlik.ipynb
├── Ders11_Dayaniklilik_ve_Dagilim_Kaymasi.ipynb
└── Ders12_Verimli_Egitim_ve_Cikarim.ipynb
```

---

## Kurulum

**Gereksinim:** Python 3.10+

```bash
# 1. Depoyu klonlayın
git clone https://github.com/HAYDARKILIC/ileri_derin_ogrenme.git
cd ileri_derin_ogrenme

# 2. Sanal ortam oluşturun (önerilir)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Bağımlılıkları kurun
pip install -r requirements.txt

# 4. Jupyter'ı başlatın
jupyter notebook
```

> Tüm defterler yalnızca NumPy, SciPy, scikit-learn ve Matplotlib ile CPU üzerinde çalışır. PyTorch,
> `requirements.txt` içinde yalnızca alıştırmaları GPU uygulamasına taşımak isteyen öğrenciler için
> listelenmiştir.

---

## Önerilen Ders Planı

| Hafta | Defterler | Tema |
|---|---|---|
| 1–3 | 01, 02, 03 | Optimizasyon, ölçek ve dikkatin maliyeti |
| 4–6 | 04, 05, 06 | Temsiller, uyarlama ve dürüst belirsizlik |
| 7–9 | 07, 08 | Üretim ve hizalama |
| 10–12 | 09, 10, 11 | Yapı, yorumlama ve başarısızlık |
| 13–14 | 12 | Sistemler ve proje çalışması |

Her defter kabaca 2–3 saatlik bir oturuma karşılık gelir: türetmeler ders, hücreler laboratuvardır.

---

## Kaynaklar

Giriş dersinden devralınan temel ders kitabı:

- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)

Defter başına seçilmiş birincil kaynaklar:

| # | Temel makaleler |
|---|---|
| 01 | Loshchilov & Hutter, *Decoupled Weight Decay Regularization*; Foret vd., *Sharpness-Aware Minimization*; Martens & Grosse, *K-FAC* |
| 02 | Kaplan vd., *Scaling Laws for Neural Language Models*; Hoffmann vd., *Training Compute-Optimal LLMs*; Shazeer vd., *Sparsely-Gated MoE*; Schaeffer vd., *Are Emergent Abilities a Mirage?* |
| 03 | Dao vd., *FlashAttention*; Shazeer, *Fast Transformer Decoding (MQA)*; Ainslie vd., *GQA*; Su vd., *RoFormer (RoPE)*; Press vd., *ALiBi*; Katharopoulos vd., *Transformers are RNNs* |
| 04 | Chen vd., *SimCLR*; Grill vd., *BYOL*; Chen & He, *SimSiam*; He vd., *MAE*; Radford vd., *CLIP*; Wang & Isola, *Alignment and Uniformity* |
| 05 | Hu vd., *LoRA*; Dettmers vd., *QLoRA*; Houlsby vd., *Adapters*; Hinton vd., *Distilling the Knowledge in a Neural Network* |
| 06 | Guo vd., *On Calibration of Modern Neural Networks*; Lakshminarayanan vd., *Deep Ensembles*; Gal & Ghahramani, *MC Dropout*; Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* |
| 07 | Ho vd., *DDPM*; Song vd., *Score-Based Generative Modeling through SDEs*; Song vd., *DDIM*; Ho & Salimans, *Classifier-Free Guidance*; Nichol & Dhariwal, *Improved DDPM* |
| 08 | Sutton & Barto, *Reinforcement Learning* (Böl. 11); van Hasselt vd., *Double DQN*; Schulman vd., *GAE*, *TRPO*, *PPO*; Ouyang vd., *InstructGPT*; Rafailov vd., *DPO*; Gao vd., *Scaling Laws for Reward Model Overoptimization* |
| 09 | Bronstein vd., *Geometric Deep Learning*; Xu vd., *How Powerful are Graph Neural Networks?*; Morris vd., *Higher-Order GNNs*; Alon & Yahav, *On the Bottleneck of GNNs* |
| 10 | Sundararajan vd., *Integrated Gradients*; Lundberg & Lee, *SHAP*; Adebayo vd., *Sanity Checks for Saliency Maps*; Elhage vd., *Toy Models of Superposition*; Bricken vd., *Towards Monosemanticity* |
| 11 | Goodfellow vd., *Explaining and Harnessing Adversarial Examples*; Madry vd., *Towards Deep Learning Models Resistant to Adversarial Attacks*; Cohen vd., *Randomized Smoothing*; Sagawa vd., *Group DRO*; Ilyas vd., *Adversarial Examples Are Not Bugs, They Are Features* |
| 12 | Micikevicius vd., *Mixed Precision Training*; Chen vd., *Training Deep Nets with Sublinear Memory Cost*; Rajbhandari vd., *ZeRO*; Shoeybi vd., *Megatron-LM*; Huang vd., *GPipe*; Leviathan vd., *Speculative Decoding* |

---

## Lisans

MIT — `LICENSE` dosyasına bakınız.
