HMERV
===============

Handwritten Mathematical Expression Recognition and Verification

Graduation Project

---------------

First stage:
---------------
preprocess image and look for related dataset

segment image into areas with printed or handwritten content 

Second stage:
---------------
1. detect printed(problems) and handwritten(solutions) texts(still need improving)

2. parse printed texts with pytesseract and recognize the handwritten

3. set up problem sets(or problem db)

4. search in the problem sets for the problem matching the parsed printed texts

5. verify the solution with the right one retrieved from problem sets

