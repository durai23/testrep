
r = np.random.normal(size=1000000)

print("The average is %.3f and the standard deviation is %.3f" % (r.mean(), r.std()))

n, b, p = hist(r, bins=100, normed=True)