library(MASS)

mu1 = c(0,0,0,0,0,0,0,0,0,0)
mu2 = c(1,1,1,1,1,1,1,1,1,1)

sigma = diag(1, 10, 10)

group1 = mvrnorm(2000, mu1, sigma)
group2 = mvrnorm(2000, mu2, sigma)

mat = matrix(c(t(group1), t(group2)), 4000, 10, byrow=TRUE)

label1 = rep(1, 2000)
label2 = rep(-1, 2000)

cc = c(label1, label2)

model = lda(mat, cc)
p = predict(model, mat)
sum(p$class == cc) / 4000

write(t(mat), "2norm.train", ncolumns=10, sep=",")

group1 = mvrnorm(200, mu1, sigma)
group2 = mvrnorm(200, mu2, sigma)

mat = matrix(c(t(group1), t(group2)), 400, 10, byrow=TRUE)
write(t(mat), "2norm.test", ncolumns=10, sep=",")