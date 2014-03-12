

import numpy as np


def get_mitschang_data():

	with open("data/mitschang-2012-t3.dat", "r") as fp:
		contents = fp.readlines()

	skip_lines = 64

	stars = {}
	all_elements = set()

	e_repr = lambda element: "[{0}/Fe]".format(element) if element != "Fe" else "[Fe/H]"

	for line in contents[skip_lines:]:
		cluster = line[:9]
		cid = line[9:12]
		star_name = line[12:25]
		sid = line[25:30]
		src = line[30:32]
		element = line[32:35]
		abundance = line[35:43]
		uncertainty = line[43:55]

		cluster, cid, star_name, sid, src, element, abundance, uncertainty \
			= [each.strip() for each in (cluster, cid, star_name, sid, src,\
				element, abundance, uncertainty)]

		# Integers:
		cid, sid = map(int, (cid, sid))
		
		float_or_nan = lambda value: float(value) if value.strip() != "" else np.nan
		abundance, uncertainty = map(float_or_nan, (abundance, uncertainty))

		# Append any unique elements to the list
		all_elements = all_elements.union({element})

		# According to the Note (2) of Table 3, we want to
		# match stars by their SID.

		# columns:
		# sid, cluster, cid, star, src, every abundance, every abundance uncertainty,
		if sid not in stars.keys():
			stars[sid] = {}

		keywords = {
			"cluster": cluster,
			"cid": cid,
			"star_name": star_name,
			"src": src,
			e_repr(element): abundance,
			"e_{0}".format(e_repr(element)): uncertainty 
		}

		# Assert that we don't have multiple abundance measurements
		# for a single star because that would be annoying
		assert element not in stars[sid]

		stars[sid].update(keywords)

	# Create a dictionary with NaN's as defaults for no measurements
	all_elements = list(all_elements)
	empty_measurements = {}
	for element in all_elements:
		
		empty_measurements[e_repr(element)] = np.nan
		empty_measurements["e_{0}".format(e_repr(element))] = np.nan
	
	# Fill in the array with nans when measurements were not available
	for sid in stars.keys():
		for key, value in empty_measurements.iteritems():
			stars[sid].setdefault(key, value)

	# Create a record array
	column_names = ("sid", "cluster", "star_name", "src", "[Na/Fe]", "e_[Na/Fe]", \
		"[Mg/Fe]", "e_[Mg/Fe]", "[Al/Fe]", "e_[Al/Fe]", "[Si/Fe]", "e_[Si/Fe]",   \
		"[Ca/Fe]", "e_[Ca/Fe]", "[Sc/Fe]", "e_[Sc/Fe]", "[Ti/Fe]", "e_[Ti/Fe]",   \
		"[V/Fe]", "e_[V/Fe]", "[Cr/Fe]", "e_[Cr/Fe]", "[Mn/Fe]", "e_[Mn/Fe]",     \
		"[Fe/H]", "e_[Fe/H]", "[Co/Fe]", "e_[Co/Fe]", "[Ni/Fe]", "e_[Ni/Fe]",     \
		"[Cu/Fe]", "e_[Cu/Fe]", "[Zn/Fe]", "e_[Zn/Fe]", "[Sr/Fe]", "e_[Sr/Fe]",   \
		"[Y/Fe]", "e_[Y/Fe]", "[Zr/Fe]", "e_[Zr/Fe]", "[Ba/Fe]", "e_[Ba/Fe]",     \
		"[La/Fe]", "e_[La/Fe]", "[Ce/Fe]", "e_[Ce/Fe]", "[Nd/Fe]", "e_[Nd/Fe]",   \
		"[Sm/Fe]", "e_[Sm/Fe]", "[Eu/Fe]", "e_[Eu/Fe]")

	data = [tuple([sid] + [stars[sid][column_name] for column_name in column_names[1:]])\
		for sid in stars.keys()]

	formats = []
	for column_name in column_names:
		if column_name.endswith("/Fe]") \
		or column_name in ("[Fe/H]", "e_[Fe/H]"):
			formats.append("f8")

		elif column_name in ("cid", "sid"):
			formats.append("i4")

		else:
			formats.append("|S15")

	return np.core.records.fromrecords(data,
		names=",".join(column_names), formats=",".join(formats))
