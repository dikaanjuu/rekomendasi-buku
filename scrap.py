import requests
import pandas as pd
import time

books = []

# =========================
# QUERY BERBAGAI TOPIK
# =========================
queries = [
    "indonesia",
    "novel indonesia",
    "cerita indonesia",
    "romance indonesia",
    "fantasy indonesia",
    "sejarah indonesia",
    "pendidikan indonesia",
    "komik indonesia",
    "sastra indonesia"
]

# =========================
# SCRAPING OPEN LIBRARY
# =========================
for q in queries:

    print(f"\nMengambil data: {q}")

    for page in range(1, 10):  # bisa dinaikkan kalau mau lebih banyak

        url = f"https://openlibrary.org/search.json?q={q}&page={page}"

        try:
            res = requests.get(url, timeout=10)

            if res.status_code != 200:
                print("❌ gagal request")
                continue

            data = res.json()

            if "docs" not in data:
                continue

            for doc in data["docs"]:

                title = doc.get("title", "")
                description = doc.get("first_sentence", "")

                # ambil bahasa jika ada
                languages = doc.get("language", [])

                lang_ok = False

                if isinstance(languages, list):
                    if "ind" in languages or "id" in languages:
                        lang_ok = True

                # fallback: tetap ambil kalau relevan indonesia
                if "indonesia" in q.lower():
                    lang_ok = True

                if lang_ok and len(title) > 2:

                    books.append({
                        "title": title,
                        "description": description if description else title
                    })

                    print("✔", title)

            time.sleep(1)

        except Exception as e:
            print("error:", e)

# =========================
# CLEAN DATA
# =========================
df = pd.DataFrame(books)

df = df.dropna()
df = df.drop_duplicates()

# filter minimal panjang teks
df = df[df["description"].str.len() > 20]

# =========================
# SAVE
# =========================
df.to_csv("books_indonesia.csv", index=False, encoding="utf-8-sig")

print("\nDONE ✔")
print("Total buku:", len(df))
print(df.head())