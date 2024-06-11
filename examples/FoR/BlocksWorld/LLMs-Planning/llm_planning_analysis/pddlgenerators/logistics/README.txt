## Logistics

* Origin: First version by Manuela Veloso, AIPS-1998 version created by Bart Selman
  and Henry Kautz. Used in both the AIPS-1998 and AIPS-2000 competitions.
* Adaptions: None.
* Description: Classical untyped STRIPS domain. Transport packages within cities via
  trucks, and between cities via airplanes. Locations within a city are directly
  connected (trucks can move between any two such locations), and so are the
  cities. In each city there is exactly one truck, each city has one location
  that serves as an airport.
* Parameters:
  * -c number of cities
  * -s size of each city, i.e. number of locations within cities
  * -p number of packages
  * -a number of airplanes
* Generation: Place trucks randomly within their cities, place airplanes randomly at
  airports. Distribute start and goal locations of packages randomly over all
  locations.

OPTIONS   DESCRIPTIONS

-a <num>    number of airplanes
-c <num>    number of cities (minimal 1)
-s <num>    city size (minimal 1)
-p <num>    number of packages (minimal 1)
-t <num>    number of trucks (optional, default and minimal: same as number of cities;
            there will be at least one truck per city)
-r <num>    random seed (minimal 1, optional)


IPC instances:

IPC00: from 4 to 15 packages, smaller instances have 2 cities, larger ones have 5 cities.
There are instances with 1 or 2 airplanes.

IPC98: cities are scaled linealy, with one more city per instance. There is an outlier
with 47 cities, all others have around 30 something. There are instances with up to 15
airplanes. Packages scale up to 57.  Sometimes there is more than one truck per city.



