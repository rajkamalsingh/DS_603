To solve this problem, we need to perform Principal Component Analysis(PCA) step - by - step based
on the scatter matrix S and given sample x.The scatter matrix S is:
𝑆 = [7.5 2.5 0
     2.5 7.5 0
     0   0   2]
(a) Determine the first principal component 𝑤 The first principal component corresponds to the eigenvector associated with the largest eigenvalue of the scatter matrix 𝑆.

Step 1: Compute the eigenvalues of 𝑆 To find the eigenvalues 𝜆, we solve the characteristic equation:

det(S−λI)=0
Substitute S and I into the equation:

det([7.5 2.5 0 - 𝜆[1 0 0 = 0
     2.5 7.5 0     0 1 0
      0   0  2]    0 0 1])


Simplify:
The determinant is calculated as:
det(S−λI)=[(7.5−λ)(7.5−λ)−(2.5)(2.5)](2−λ)
Simplify the first term:
(7.5−λ) ^ 2 −2.5 ^ 2 = (7.5−λ) ^ 2 −6.25
Thus, the determinant becomes:
[(7.5−λ) ^ 2−6.25](2−λ) = 0
Step 2: Solve for the eigenvalues

Step 3: Find the eigenvector corresponding to𝜆1 = 10
The first principal component 𝑤 corresponds to the eigenvector associated with the largest eigenvalue 𝜆1=10
To find this eigenvector, solve:
(S−10I)w = 0
SubstituteS and 10I:
Thus, the eigenvector is:

𝑤 = [1
     1
     0]
Normalize 𝑤 to have unit length:
| | w | |= sqrt(1 ^ 2 + 1 ^ 2 + 0 ^ 2)

Therefore, the first principal component is:

𝑤 = | | w | |.w

(b)
Approximate x using the first principal component The goal is to project x = [1.2, 1.7,−1.3] ^ T onto
the line spanned by w.The projection of x onto w is given by:

𝑥~ = (𝑤 ^ 𝑇.𝑥)𝑤





