"""
NeuroCausal RAG - Türkçe İklim Değişikliği Bilgi Tabanı
115+ döküman (Sadece ham veri - nedensel ilişki YOK!)

ÖNEMLİ: Bu dosya SADECE dökümanları içerir.
        Nedensel ilişkiler MANUEL tanımlanmaz!
        AutoCausalDiscovery modülü bunları OTOMATİK keşfeder.

Yazar: Ertugrul Akben
E-posta: i@ertugrulakben.com
Versiyon: 4.0.0
"""

# =============================================================================
# DÖKÜMANLAR - 100+ Türkçe içerik
# =============================================================================
DOCUMENTS = [
    # TEMEL KAVRAMLAR (1-15)
    {"id": "iklim_tanimi", "content": "İklim, bir bölgede uzun yıllar boyunca gözlemlenen ortalama hava koşullarıdır. Sıcaklık, yağış, nem ve rüzgar gibi atmosferik olayların uzun vadeli ortalamasını ifade eder.", "category": "temel"},
    {"id": "iklim_degisikligi", "content": "İklim değişikliği, iklim örüntülerindeki uzun vadeli değişimleri ifade eder. İnsan faaliyetlerinden kaynaklanan sera gazı emisyonları nedeniyle hızlanmıştır.", "category": "temel"},
    {"id": "kuresel_isinma", "content": "Küresel ısınma, Dünya'nın ortalama yüzey sıcaklığının artmasıdır. Son 100 yılda yaklaşık 1.1°C artış gözlemlenmiştir.", "category": "temel"},
    {"id": "sera_etkisi", "content": "Sera etkisi, atmosferdeki gazların güneş ışınlarını tutarak Dünya'nın ısınmasına neden olmasıdır. Bu doğal bir süreçtir ancak insan faaliyetleri ile güçlenmiştir.", "category": "temel"},
    {"id": "sera_gazlari", "content": "Sera gazları atmosferde ısıyı tutan gazlardır. Başlıcaları karbondioksit (CO2), metan (CH4), su buharı ve diazot monoksittir (N2O).", "category": "temel"},
    {"id": "co2_tanimi", "content": "Karbondioksit (CO2), fosil yakıtların yanması ve ormansızlaşma sonucu atmosfere salınan en önemli sera gazıdır.", "category": "temel"},
    {"id": "metan_tanimi", "content": "Metan (CH4), CO2'den 80 kat daha güçlü bir sera gazıdır. Hayvancılık, pirinç tarımı ve doğalgaz sızıntılarından kaynaklanır.", "category": "temel"},
    {"id": "karbon_dongusu", "content": "Karbon döngüsü, karbonun atmosfer, okyanuslar, bitkiler ve toprak arasındaki dolaşımıdır. İnsan faaliyetleri bu döngüyü bozmuştur.", "category": "temel"},
    {"id": "atmosfer_yapisi", "content": "Atmosfer, Dünya'yı saran gaz katmanıdır. Troposfer, stratosfer ve diğer katmanlardan oluşur. İklim olayları troposferde gerçekleşir.", "category": "temel"},
    {"id": "gunes_radyasyonu", "content": "Güneş radyasyonu, Dünya'nın enerji kaynağıdır. Bir kısmı yeryüzünden yansır, bir kısmı sera gazları tarafından tutulur.", "category": "temel"},
    {"id": "albedo_etkisi", "content": "Albedo, bir yüzeyin güneş ışığını yansıtma oranıdır. Kar ve buz yüksek albedoya sahiptir. Eridiklerinde daha fazla enerji emilir.", "category": "temel"},
    {"id": "okyanuslar_rol", "content": "Okyanuslar, atmosferdeki CO2'nin yaklaşık %30'unu emer ve ısının %90'ından fazlasını depolar. İklim düzenlemesinde kritik rol oynar.", "category": "temel"},
    {"id": "biyosfer", "content": "Biyosfer, tüm canlıların yaşadığı alandır. Ormanlar, okyanuslar ve toprak ekosistemlerini kapsar. Karbon depolama işlevi görür.", "category": "temel"},
    {"id": "kriyosfer", "content": "Kriyosfer, Dünya'nın buzla kaplı bölgelerini ifade eder. Kutup buzulları, dağ buzulları ve permafrost buna dahildir.", "category": "temel"},
    {"id": "hidrosfer", "content": "Hidrosfer, Dünya'daki tüm su kaynaklarını kapsar. Okyanuslar, göller, nehirler ve yer altı suları bu sistemin parçasıdır.", "category": "temel"},

    # NEDENLER (16-35)
    {"id": "fosil_yakit", "content": "Fosil yakıtlar (kömür, petrol, doğalgaz) yanınca CO2 açığa çıkar. Enerji üretiminin %80'i fosil yakıtlardan sağlanır.", "category": "neden"},
    {"id": "komur_yanmasi", "content": "Kömür yanması en kirli enerji kaynağıdır. Megawatt başına en yüksek CO2 emisyonuna neden olur.", "category": "neden"},
    {"id": "petrol_tuketimi", "content": "Petrol, ulaşım sektörünün temel yakıtıdır. Taşıtlar küresel CO2 emisyonlarının %16'sından sorumludur.", "category": "neden"},
    {"id": "dogalgaz", "content": "Doğalgaz, kömürden temiz ancak yine de fosil yakıttır. Sızıntıları metan emisyonuna neden olur.", "category": "neden"},
    {"id": "ormansizlasma", "content": "Ormansızlaşma, karbon yutak kapasitesini azaltır ve depolanan karbonu atmosfere salar. Yılda 10 milyon hektar orman kaybedilir.", "category": "neden"},
    {"id": "tarim_emisyon", "content": "Tarım sektörü sera gazı emisyonlarının %10-12'sinden sorumludur. Gübre kullanımı, pirinç tarımı ve toprak işleme başlıca kaynaktır.", "category": "neden"},
    {"id": "hayvancilik", "content": "Hayvancılık, özellikle sığır yetiştiriciliği, metan emisyonlarının önemli kaynağıdır. Sindirim süreçleri metan üretir.", "category": "neden"},
    {"id": "sanayi_devrim", "content": "Sanayi Devrimi (1750) ile fosil yakıt kullanımı başladı. O tarihten bu yana CO2 konsantrasyonu %50 arttı.", "category": "neden"},
    {"id": "ulasim", "content": "Ulaşım sektörü küresel emisyonların %16'sını oluşturur. Karayolu taşımacılığı en büyük paya sahiptir.", "category": "neden"},
    {"id": "havacilik", "content": "Havacılık sektörü hızla büyüyor ve emisyonların %2.5'ini oluşturuyor. Yüksek irtifada salınan gazlar daha zararlıdır.", "category": "neden"},
    {"id": "deniz_tasimaciligi", "content": "Deniz taşımacılığı küresel ticaretin %80'ini taşır ve emisyonların %3'ünü oluşturur.", "category": "neden"},
    {"id": "bina_isitma", "content": "Binaların ısıtılması ve soğutulması enerji tüketiminin %40'ını oluşturur. Çoğu fosil yakıtla çalışır.", "category": "neden"},
    {"id": "cimento_uretimi", "content": "Çimento üretimi küresel CO2 emisyonlarının %8'inden sorumludur. Kireçtaşının ısıtılması sırasında CO2 açığa çıkar.", "category": "neden"},
    {"id": "demir_celik", "content": "Demir-çelik sektörü enerji yoğun bir sektördür ve emisyonların %7'sini oluşturur.", "category": "neden"},
    {"id": "nufus_artisi", "content": "Dünya nüfusu 8 milyara ulaştı. Daha fazla insan, daha fazla enerji ve kaynak tüketimi demektir.", "category": "neden"},
    {"id": "tuketim_aliskanliklari", "content": "Modern tüketim alışkanlıkları sera gazı emisyonlarını artırır. Hızlı moda, tek kullanımlık ürünler örnek verilebilir.", "category": "neden"},
    {"id": "plastik_uretimi", "content": "Plastik üretimi petrol bazlıdır ve emisyonlara katkıda bulunur. Ayrıca çöp olarak uzun süre doğada kalır.", "category": "neden"},
    {"id": "gida_israf", "content": "Üretilen gıdaların %30'u israf edilir. İsraf edilen gıda, üretim ve taşıma emisyonlarını boşa harcar.", "category": "neden"},
    {"id": "kentlesme", "content": "Kentleşme arazi kullanımını değiştirir, karbon yutaklarını azaltır ve enerji tüketimini artırır.", "category": "neden"},
    {"id": "permafrost_erime", "content": "Permafrost eridikçe içindeki metan ve CO2 atmosfere salınır. Bu pozitif geri besleme döngüsü oluşturur.", "category": "neden"},

    # SONUÇLAR (36-65)
    {"id": "sicaklik_artisi", "content": "Küresel ortalama sıcaklık son 100 yılda 1.1°C arttı. Bu artış dengesiz dağılır, kutuplar daha fazla ısınır.", "category": "sonuc"},
    {"id": "buzul_erimesi", "content": "Buzullar hızla eriyor. Grönland ve Antarktika'dan yılda 270 milyar ton buz kaybediliyor.", "category": "sonuc"},
    {"id": "deniz_seviyesi", "content": "Deniz seviyesi 1900'den bu yana 20 cm yükseldi. 2100'e kadar 1 metreye kadar yükselebilir.", "category": "sonuc"},
    {"id": "asiri_hava", "content": "Aşırı hava olayları (fırtınalar, seller, kuraklıklar) sıklaşıyor ve şiddetleniyor.", "category": "sonuc"},
    {"id": "kuraklik", "content": "Kuraklıklar daha uzun ve şiddetli hale geliyor. Akdeniz havzası özellikle risk altındadır.", "category": "sonuc"},
    {"id": "sel_baskini", "content": "Sel baskınları sıklaşıyor. Şiddetli yağışlar altyapıyı zorluyor.", "category": "sonuc"},
    {"id": "sicak_dalgasi", "content": "Sıcak dalgaları daha sık ve şiddetli. 2022 Avrupa sıcak dalgasında 60.000'den fazla kişi öldü.", "category": "sonuc"},
    {"id": "yangin_artisi", "content": "Orman yangınları sıklaşıyor. Avustralya, Kaliforniya ve Akdeniz bölgesi yıkıcı yangınlar yaşadı.", "category": "sonuc"},
    {"id": "kasirga_guc", "content": "Kasırgalar daha güçleniyor. Sıcak okyanuslar daha fazla enerji sağlar.", "category": "sonuc"},
    {"id": "okyanus_asitlenmesi", "content": "Okyanuslar CO2 emdikçe asitleniyor. pH seviyesi 0.1 düştü. Mercan ve kabuklu deniz canlıları risk altında.", "category": "sonuc"},
    {"id": "mercan_agarmasi", "content": "Sıcak sular mercanlarda ağarmaya neden olur. Büyük Set Resifi'nin %50'si zarar gördü.", "category": "sonuc"},
    {"id": "biyocesitlilik_kaybi", "content": "Türlerin %1 milyonu yok olma tehlikesiyle karşı karşıya. İklim değişikliği habitatları yok eder.", "category": "sonuc"},
    {"id": "kutup_ayisi", "content": "Kutup ayıları buz kaybı nedeniyle tehlike altında. Av alanları daraldı.", "category": "sonuc"},
    {"id": "goc_yollari", "content": "Kuş göç yolları ve zamanlamaları değişiyor. Bazı türler uyum sağlayamıyor.", "category": "sonuc"},
    {"id": "tarim_verim", "content": "Tarım verimliliği bazı bölgelerde düşüyor. Kuraklık ve sıcaklık ürün kaybına yol açıyor.", "category": "sonuc"},
    {"id": "gida_guvenlik", "content": "Gıda güvenliği tehdit altında. İklim şokları arz ve fiyatları etkiliyor.", "category": "sonuc"},
    {"id": "su_kitligi", "content": "Su kıtlığı artıyor. 2 milyar insan su stresi altında yaşıyor.", "category": "sonuc"},
    {"id": "saglik_etki", "content": "İklim değişikliği sağlığı etkiler: sıcak stresi, hava kirliliği, bulaşıcı hastalıklar yaygınlaşır.", "category": "sonuc"},
    {"id": "sivrisinek_hastalik", "content": "Sivrisinek kaynaklı hastalıklar (sıtma, dang) yeni bölgelere yayılıyor.", "category": "sonuc"},
    {"id": "ekonomik_zarar", "content": "İklim değişikliğinin ekonomik maliyeti trilyon dolarlarla ifade ediliyor. 2050'ye kadar GSYİH'nin %2-4'ü kaybedilebilir.", "category": "sonuc"},
    {"id": "iklim_gocmeni", "content": "İklim göçmenleri artıyor. 2050'ye kadar 200 milyon kişi yerinden edilebilir.", "category": "sonuc"},
    {"id": "altyapi_hasar", "content": "Altyapı hasarı artıyor. Yollar, köprüler ve binalar aşırı hava olaylarından etkilenir.", "category": "sonuc"},
    {"id": "sigorta_maliyet", "content": "Sigorta maliyetleri yükseliyor. Doğal afet kayıpları rekor kırıyor.", "category": "sonuc"},
    {"id": "tarim_bolge", "content": "Tarım bölgeleri kuzeye kayıyor. Geleneksel ürünler bazı bölgelerde yetişmiyor.", "category": "sonuc"},
    {"id": "ormanlık_alan", "content": "Ormanlık alanlar değişiyor. Yangınlar ve zararlılar orman yapısını etkiliyor.", "category": "sonuc"},
    {"id": "tatlisu_kaynak", "content": "Tatlı su kaynakları azalıyor. Buzullar eriyor ve yağış desenleri değişiyor.", "category": "sonuc"},
    {"id": "balikcilik", "content": "Balıkçılık etkileniyor. Balık stokları göç ediyor ve azalıyor.", "category": "sonuc"},
    {"id": "turizm_etki", "content": "Turizm etkileniyor. Kayak merkezleri, sahil bölgeleri ve doğal alanlar risk altında.", "category": "sonuc"},
    {"id": "enerji_talep", "content": "Soğutma için enerji talebi artıyor. Sıcak havada klima kullanımı emisyonları artırır.", "category": "sonuc"},
    {"id": "kentsel_isi", "content": "Kentsel ısı adası etkisi güçleniyor. Şehirler çevrelerinden 2-3°C daha sıcak olabiliyor.", "category": "sonuc"},

    # ÇÖZÜMLER (66-90)
    {"id": "yenilenebilir_enerji", "content": "Yenilenebilir enerji (güneş, rüzgar, hidro) fosil yakıtlara temiz alternatiftir. Maliyetler hızla düşüyor.", "category": "cozum"},
    {"id": "gunes_enerjisi", "content": "Güneş enerjisi en hızlı büyüyen enerji kaynağı. Solar panel maliyeti %90 düştü.", "category": "cozum"},
    {"id": "ruzgar_enerjisi", "content": "Rüzgar enerjisi önemli bir potansiyele sahip. Karasal ve deniz üstü rüzgar çiftlikleri yaygınlaşıyor.", "category": "cozum"},
    {"id": "nukleer_enerji", "content": "Nükleer enerji düşük karbonlu elektrik üretir ancak güvenlik ve atık sorunları vardır.", "category": "cozum"},
    {"id": "elektrikli_arac", "content": "Elektrikli araçlar ulaşım emisyonlarını azaltır. Batarya teknolojisi hızla gelişiyor.", "category": "cozum"},
    {"id": "toplu_tasima", "content": "Toplu taşıma bireysel araç kullanımından daha az emisyona neden olur.", "category": "cozum"},
    {"id": "bisiklet_yurume", "content": "Bisiklet ve yürüme sıfır emisyonlu ulaşım sağlar ve sağlığı destekler.", "category": "cozum"},
    {"id": "enerji_verimlilik", "content": "Enerji verimliliği enerji tüketimini azaltır. LED aydınlatma, yalıtım önemli önlemlerdir.", "category": "cozum"},
    {"id": "yesil_bina", "content": "Yeşil binalar sürdürülebilir tasarım prensipleriyle inşa edilir. Enerji tüketimini %50'ye kadar azaltabilir.", "category": "cozum"},
    {"id": "agaclandirma", "content": "Ağaçlandırma karbon tutma kapasitesi sağlar. 1 hektar orman yılda 10 ton CO2 emer.", "category": "cozum"},
    {"id": "orman_koruma", "content": "Orman koruma mevcut karbon stoklarını korur. REDD+ programları bunu teşvik eder.", "category": "cozum"},
    {"id": "karbon_yakalama", "content": "Karbon yakalama ve depolama (CCS) teknolojisi CO2'yi kaynağında yakalayıp yer altında depolar.", "category": "cozum"},
    {"id": "hidrojen_enerji", "content": "Yeşil hidrojen temiz enerji taşıyıcısı olabilir. Ağır sanayi ve ulaşımda kullanılabilir.", "category": "cozum"},
    {"id": "surdurulebilir_tarim", "content": "Sürdürülebilir tarım toprak sağlığını korur ve emisyonları azaltır. Organik tarım bir yöntemdir.", "category": "cozum"},
    {"id": "bitkisel_beslenme", "content": "Bitkisel ağırlıklı beslenme hayvancılık kaynaklı emisyonları azaltır.", "category": "cozum"},
    {"id": "gida_israf_onleme", "content": "Gıda israfını önleme emisyonları ve kaynak kullanımını azaltır.", "category": "cozum"},
    {"id": "dongusel_ekonomi", "content": "Döngüsel ekonomi atığı minimuma indirir ve kaynakları yeniden kullanır.", "category": "cozum"},
    {"id": "karbon_fiyatlama", "content": "Karbon fiyatlandırma (vergi veya emisyon ticareti) piyasa teşviki sağlar.", "category": "cozum"},
    {"id": "paris_anlasması", "content": "Paris İklim Anlaşması ısınmayı 1.5°C ile sınırlamayı hedefler. 195 ülke imzaladı.", "category": "cozum"},
    {"id": "yerel_eylem", "content": "Yerel eylemler küresel çözüme katkıda bulunur. Belediyeler iklim planları hazırlıyor.", "category": "cozum"},
    {"id": "egitim_farkindalik", "content": "İklim eğitimi ve farkındalık davranış değişikliği sağlar.", "category": "cozum"},
    {"id": "inovasyon", "content": "Temiz teknoloji inovasyonu çözümler için kritik önemdedir.", "category": "cozum"},
    {"id": "finansman", "content": "İklim finansmanı gelişmekte olan ülkelerin adaptasyonunu destekler.", "category": "cozum"},
    {"id": "uyum_stratejisi", "content": "Adaptasyon stratejileri iklim etkilerine uyum sağlamayı hedefler.", "category": "cozum"},
    {"id": "erken_uyari", "content": "Erken uyarı sistemleri aşırı hava olaylarından korunmayı sağlar.", "category": "cozum"},

    # BİLİMSEL ÖLÇÜMLER (91-105)
    {"id": "co2_ppm", "content": "Atmosferik CO2 konsantrasyonu 420 ppm'e ulaştı. Sanayi öncesi dönemde 280 ppm'di.", "category": "olcum"},
    {"id": "ipcc_rapor", "content": "IPCC (Hükümetlerarası İklim Değişikliği Paneli) iklim biliminin en yetkili kaynağıdır.", "category": "olcum"},
    {"id": "iklim_modeli", "content": "İklim modelleri gelecek senaryolarını simüle eder. Farklı emisyon yolları incelenir.", "category": "olcum"},
    {"id": "buz_karot", "content": "Buz karotları geçmiş iklimi anlamaya yardımcı olur. 800.000 yıllık veri elde edildi.", "category": "olcum"},
    {"id": "sicaklik_olcum", "content": "Küresel sıcaklık ölçümleri 1850'den beri yapılıyor. Uydu verileri 1979'dan beri mevcut.", "category": "olcum"},
    {"id": "deniz_olcum", "content": "Deniz seviyesi tide gauge ve uydu altimetrisi ile ölçülür.", "category": "olcum"},
    {"id": "emisyon_envanter", "content": "Ulusal emisyon envanterleri ülkelerin sera gazı salımlarını izler.", "category": "olcum"},
    {"id": "karbon_ayak_izi", "content": "Karbon ayak izi bireylerin veya kuruluşların emisyon miktarını ölçer.", "category": "olcum"},
    {"id": "rcp_senaryo", "content": "RCP senaryoları farklı emisyon yollarını temsil eder. RCP2.6'dan RCP8.5'e kadar değişir.", "category": "olcum"},
    {"id": "ssp_senaryo", "content": "SSP senaryoları sosyoekonomik varsayımları içerir. İklim projeksiyonlarında kullanılır.", "category": "olcum"},
    {"id": "karbon_butce", "content": "Karbon bütçesi 1.5°C hedefi için kalan emisyon payını ifade eder. Hızla tükeniyor.", "category": "olcum"},
    {"id": "net_sifir", "content": "Net sıfır, emisyonların tutma ile dengelenmesidir. 2050 hedefi yaygın kabul görüyor.", "category": "olcum"},
    {"id": "tipping_point", "content": "Devrilme noktaları geri dönüşü olmayan değişimleri ifade eder. Amazon, buzullar örnek verilebilir.", "category": "olcum"},
    {"id": "geri_besleme", "content": "Geri besleme döngüleri ısınmayı güçlendirebilir veya zayıflatabilir.", "category": "olcum"},
    {"id": "iklim_duyarlılık", "content": "İklim duyarlılığı CO2 iki katına çıktığında sıcaklık artışını ifade eder. 2.5-4°C arasında tahmin ediliyor.", "category": "olcum"},

    # TÜRKİYE ÖZELİNDE (106-115)
    {"id": "turkiye_iklim", "content": "Türkiye Akdeniz iklim bölgesinde özellikle risk altındadır. Sıcaklık 1.5°C arttı.", "category": "turkiye"},
    {"id": "turkiye_kuraklik", "content": "Türkiye'de kuraklık yaygınlaşıyor. Konya Ovası ve İç Anadolu su stresi altında.", "category": "turkiye"},
    {"id": "turkiye_sel", "content": "Karadeniz bölgesinde şiddetli sel olayları artıyor. 2021 Kastamonu seli örnek.", "category": "turkiye"},
    {"id": "turkiye_sicak", "content": "Türkiye'de sıcak dalgaları sıklaşıyor. 2021'de 49.1°C rekor kırıldı.", "category": "turkiye"},
    {"id": "turkiye_yangin", "content": "Orman yangınları artıyor. 2021 Akdeniz yangınlarında 160.000 hektar yandı.", "category": "turkiye"},
    {"id": "turkiye_tarim", "content": "Türk tarımı iklim değişikliğinden etkileniyor. Buğday ve fındık verimi düşebilir.", "category": "turkiye"},
    {"id": "turkiye_enerji", "content": "Türkiye enerji ithalatına bağımlı. Yenilenebilir enerji potansiyeli yüksek.", "category": "turkiye"},
    {"id": "turkiye_paris", "content": "Türkiye Paris Anlaşması'nı 2021'de onayladı. 2053 net sıfır hedefi açıkladı.", "category": "turkiye"},
    {"id": "turkiye_gunes", "content": "Türkiye güneş enerjisi potansiyeli yüksek ülkelerden biri. Kapasitesi hızla artıyor.", "category": "turkiye"},
    {"id": "turkiye_ruzgar", "content": "Türkiye'nin rüzgar enerjisi kapasitesi 10 GW'ı aştı. Ege ve Marmara bölgeleri öncü.", "category": "turkiye"},
]


# =============================================================================
# KULLANIM
# =============================================================================
def get_documents():
    """Tüm dökümanları döndür"""
    return DOCUMENTS


def get_documents_by_category(category: str):
    """Kategoriye göre dökümanları filtrele"""
    return [d for d in DOCUMENTS if d.get("category") == category]


def get_document_count():
    """Döküman sayısını döndür"""
    return len(DOCUMENTS)


if __name__ == "__main__":
    print(f"Toplam döküman: {get_document_count()}")
    print("\nKategoriler:")
    categories = set(d.get("category") for d in DOCUMENTS)
    for cat in sorted(categories):
        count = len(get_documents_by_category(cat))
        print(f"  - {cat}: {count} döküman")
    print("\nNOT: Nedensel ilişkiler TANIMLANMADI!")
    print("      AutoCausalDiscovery modülü bunları OTOMATİK keşfeder.")
