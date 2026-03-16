# NeuroCausal RAG - Hizli Baslangic

5 dakikada sistemi test edin!

---

## 1. Playground'a Gidin

Yan menuden **Playground** sayfasini secin.

---

## 2. Ornek Veri Yukleyin

### Secenek A: Hazir Ornek
"Ornek Yukle" butonuna tiklayin. Otomatik olarak iklim verileri yuklenir.

### Secenek B: Kendi Veriniz
Metin kutusuna kendi verilerinizi yapistirin.

Ornek format:
```
### belge_adi
Belge icerigi burada. Cumleler nedensellik iliskisi icermelidir.

### baska_belge
Baska bir belge icerigi. Sistem bunlar arasindaki iliskileri bulur.
```

---

## 3. Analiz Edin

"Analiz Et" butonuna basin. Sistem:
- Dokumanlari embedding'e cevirir
- Nedensellik iliskilerini kesfeder
- Graf yapisini olusturur

Ilerleme cubugu sureci gosterir.

---

## 4. Soru Sorun

Sorgu kutusuna bir soru yazin:
- "Sera gazlari nelerdir?"
- "Iklim degisikliginin sonuclari nelerdir?"
- "Cimento uretimi iklimi nasil etkiler?"

---

## 5. Sonuclari Inceleyin

Sistem size gosterir:
- **Bulunan Dokumanlar:** Ilgili sonuclar
- **Nedensellik Zinciri:** A -> B -> C baglantilari
- **Skor Detaylari:** Benzerlik + Nedensellik + Onem

---

## Demo Senaryolari

### Senaryo 1: Iklim Zinciri
```
Fosil yakitlar -> CO2 -> Sera etkisi -> Isinma -> Buzul erimesi
```

### Senaryo 2: Stres Zinciri
```
Stres -> Kortizol -> Uyku bozuklugu -> Dikkat -> Kaza
```

### Senaryo 3: Ekonomi Zinciri
```
Faiz artisi -> Kredi daralma -> Tuketim azalma -> Issizlik
```

---

## Ipuclari

1. **Cumleler net olsun:** "A, B'ye neden olur" gibi ifadeler daha iyi calisiyor
2. **Zincir olusturun:** Birbirine baglanan dokumanlar daha iyi sonuc verir
3. **Sorulari test edin:** Farkli sorularla sistemi test edin

---

## SSS

**S: Minimum kac dokuman gerekli?**
C: En az 3-5 dokuman onerilir.

**S: Maksimum kac dokuman yukleyebilirim?**
C: Playground'da ~50 dokuman. Production'da sinir yok.

**S: Turkce mi Ingilizce mi?**
C: Her ikisi de calisir. Embedding modeli cok dilli.

**S: Verilerim kaydediliyor mu?**
C: Hayir, Playground gecici (in-memory) calisir.

---

## Sonraki Adimlar

1. **Kendi verinizi deneyin:** Is verilerinizi yukleyin
2. **API kullanin:** `/api/v1/search` endpoint'i
3. **Production'a gecin:** Docker ile deploy edin

---

**Yardim:** Sorulariniz icin GitHub Issues kullanin.
