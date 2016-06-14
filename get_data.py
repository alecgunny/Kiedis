from six.moves import urllib
#import urllib2
from bs4 import BeautifulSoup as bs

def parse_lyrics(html):
	lyrics = ''
	go = False
	Chorus = ''
	adding_to_chorus = False
	for line in html.split("\n"):
		if "Usage of azlyrics.com" in line:
			go = True
			continue
		if not go:
			continue
		if "</div>" in line:
			break
		if line == "<br>" and not adding_to_chorus:
			continue
		elif line == "<br>":
			adding_to_chorus = False
			continue
		if 'Chorus' in line or 'Refrain' in line and len(Chorus) == 0:
			adding_to_chorus = True
			continue
		elif 'Chorus' in line or 'Refrain':
			lyrics += Chorus
			continue
		elif line.startswith('<i>'):
			continue

		lyrics += line.split("<br>")[0] + " "
		if adding_to_chorus:
			Chorus += line.split("<br>")[0] + " "
	return lyrics


def get_all_lyrics(home_page='http://www.azlyrics.com/r/redhotchilipeppers.html'):
	# opener = urllib2.build_opener()

	# headers = {
	#   'User-Agent': 'Mozilla/5.0 (Windows NT 5.1; rv:10.0.1) Gecko/20100101 Firefox/10.0.1',
	# }

	# opener.addheaders = headers.items()
	# home = opener.open(home_page)
	home = urllib.request.urlopen(home_page)
	soup = bs(home)
	lyrics = ''
	for link in soup.findAll('a'):
		href = link.get('href')
		if href is None:
			continue
		if 'lyrics/redhotchilipeppers' not in href:
			continue
		new_url = 'http://www.azlyrics.com' + href[2:]
		print "Mining url: %s" % new_url
		try:
			# lyrics_page = opener.open(new_url).read()
			lyrics_page = urllib.request.urlopen(new_url).read()
			lyrics += parse_lyrics(lyrics_page) + " "
		except:
			print "Encountered error on url %s, skipping..." % new_url
	return lyrics

if __name__ == "__main__":
	lyrics = get_all_lyrics()
	with open('rhcp_lyrics.txt', 'w') as f:
		f.write(lyrics)
