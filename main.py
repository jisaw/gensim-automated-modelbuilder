from gensim import corpora, models
import gensim_utility as gu
import sys
import timeit


def build_lsi(d, c, fname, n, tfidf=False):
	start = timeit.default_timer()
	print("[+] Loading dictionary and corpus")
	tfidf_corpus = tfidf[c] if tfidf else c
	lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=n)
	if tfidf:
		lsi.save("nhtsa/" + fname + "-" + str(n) + '-tfidf.lsi')
	else:
		lsi.save("nhtsa/" + fname + "-" + str(n) + '.lsi')
	print("[+] Model Saved")
	stop = timeit.default_timer()
	with open('nhtsa/fname-'+str(n)+".txt", 'a') as f:
		print >> f, "Operation Time in Epoch:%s" % stop-start
		for i in lsi.print_topics(n):
			print >> f, str(i)


def build_lda(d, c, fname, n, tfidf=False):
	start = timeit.default_timer()
	print("[+] Loading dictionary and corpus")
	tfidf_corpus = tfidf[c] if tfidf else c
	lda = model.LdaModel(tfidf_corpus, id2word=dictionary, num_topics=n)
	if tfidf:
		lda.save("nhtsa/" + fname + "-" + str(n) + '-tfidf.lda')
	else:
		lda.save("nhtsa/" + fname + "-" + str(n) + '.lda')
	print("[+] Model Saved")
	stop = timeit.default_timer()
	with open('nhtsa/fname-'+str(n)+".txt", 'a') as f:
		print >> f, "Operation Time in Epoch:%s" % stop-start
		for i in lda.print_topics(n):
			print >> f, str(i)

def main():
	print('[+] Loading Dependent Resources')
	d = corpora.Dictionary.load(str(sys.argv[1]))
	c = corpora.MmCorpus(str(sys.argv[2]))
	fname = str(sys.argv[3])
	tfidf = models.TfidfModel.load(str(sys.argv[4]))
	for n in range(3,13):
		print("[+] Starting Operation %s" % n-2)
		#Vanilla
		build_lsi(d,c,fname,n)
		print("[+] Vanilla Lsi finished")
		build_lda(d,c,fname,n)
		print("[+] Vanilla lda finished")
		#w/ Tfidf
		build_lsi(d,c,fname,n.tfidf)
		print("[+] Tfidf Lsi finished")
		build_lda(d,c,fname,n.tfidf)
		print("[+] Tfidf Lda finished")

if __name__ == "__main__":
	main()