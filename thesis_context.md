Temat:
Automat dzielący polski tekst prawny na zdania z wykorzystaniem mechanizmów uczenia maszynowego Temat po angielsku:
Automatic division of Polish legal text into sentences using machine learning mechanisms
Cel: Polskie teksty prawne, np. ustawy, rozporządzenia czy orzeczenia sądowe, napisane są w specyficznym języku prawniczym, pełnym skrótów czy wypunktowań. Dostępne mechanizmy podziału tekstów w języku polskim na zdania (spacy, NLTK) popełniają błędy w przypadku tekstów prawnych. Celem pracy jest (1) dogłębna analiza istniejących rozwiązań realizujących podział tekstu na zdania ze szczególnym uwzględnieniem j. polskiego, (2) przetestowanie wybranych rozwiązań open-source na tekstach prawnych i ich ocena według przyjętych kryteriów, (3) douczenie wybranego modelu sentencizera do lepszej realizacji podziału na zdania tekstów prawnych. Charakter: projektowa Charakter po angielsku: project work Typ dyplomu Standardowa praca dyplomowa Data wydania dyplomu: Komentarz: Zadania do wykonania:
(1) dogłębna analiza istniejących rozwiązań realizujących podział tekstu na zdania ze szczególnym uwzględnieniem j. polskiego
(2) przetestowanie wybranych rozwiązań open-source na tekstach prawnych (ogólnodostępnych aktów prawnych i orzeczeń sądowych)
(3) punkt (2) będzie wymagał wyciągania strumieni tekstowych z dokumentów różnych formatów - głównie PDF, DOCX
(4) ustalenie kryteriów oceny poprawności testowanych rozwiązań
(5) douczenie wybranego modelu do lepszej realizacji podziału na zdania tekstów prawnych (skuteczność potwierdzona testami i obliczona wg kryteriów ustalonych w (4) Literatura:
Dziennik ustaw https://www.dziennikustaw.gov.pl/DU
Portal orzeczeń sądów powszechnych https://orzeczenia.ms.gov.pl/
Speech and Language Processing, Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/).
Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation, Markus Frohmann et.al, 2024. https://arxiv.org/abs/2406.16678
Sentence Boundary Extraction from Scientific Literature of Electric Double Layer Capacitor Domain: Tools and Techniques, Miah, M.S.U. et.al. Appl. Sci. 2022, 12, 1352. https://doi.org/10.3390/app12031352
https://spacy.io/api/sentencizer/
www.nltk.org/api/nltk.tokenize.sent_tokenize.html