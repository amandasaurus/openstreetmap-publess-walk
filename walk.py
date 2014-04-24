#! /usr/bin/python

from xml.etree.ElementTree import ElementTree
from collections import defaultdict
import itertools, simplejson, math
from heapq import heappop, heappush
import urllib
from contextlib import contextmanager
from pprint import pprint
from copy import deepcopy
import os.path


# in metres
EARTH_RADIUS = 6367 * 1000

def is_pub(node):
    return node['tags'].get("amenity") == 'pub'

def parse_osm_file(osm_file):
    """
    Given an OSM filename ``osm_file``
    """
    nodes = {}
    ways = {}
    nodes_ways = defaultdict(list)
    ways_ways = defaultdict(dict)
    node_connections = defaultdict(set)
    pubs = set()

    tree = ElementTree()
    tree.parse(osm_file)
    with print_status("Getting nodes... "):
        for node in tree.getiterator("node"):
            n = node.attrib
            n['lat'] = float(n['lat'])
            n['lon'] = float(n['lon'])
            n['id'] = int(n['id'])
            n['tags'] = dict((x.attrib['k'], x.attrib['v']) for x in node.findall("tag"))
            nodes[int(n['id'])] = n
            if is_pub(n):
                pubs.add(n['id'])

    with print_status("Getting ways... "):
        for way in tree.getiterator("way"):
            w = way.attrib
            w['id'] = int(w['id'])
            w['tags'] = dict((x.attrib['k'], x.attrib['v']) for x in way.findall("tag"))
            nds = [int(x.attrib['ref']) for x in way.findall("nd")]
            for nd in nds:
                nodes_ways[nd].append(w['id'])


            w['nodes'] = [nodes[x] for x in nds]
            connect_nodes_in_way(w, node_connections)
            ways[w['id']] = w

    return nodes, ways, nodes_ways, ways_ways, node_connections, pubs

def one_side_of_dublin_to_the_other():
    """
    Returns a list 2-tuples of node ids. Each tuple is 2 nodes that are across
    dublin from each other. They are based on the Dublin Outer Orbital Route
    """
    # starting at the east link toll bridge, note: order is important.
    southside_canal_nodes = [
        30978986, 30979098, 31093638, 32334985, 32335036, 32335290,
        9100868, 29400040, 29396438, 32336034, 32336040, 4238407,
        32336199, 12784893,
    ]
    #southside_canal_nodes = [ 32335006 ]

    # starts in islandbridge (nr. phoenix park) and goes towards the point, note: order is important
    northside_canal_nodes = [
        12784890, 12428414, 117707333, 12246878, 11675220,
        659686, 659687, 659706, 26765400, 20447294, 20299851,
        12117928, 389372, 27417734,
    ]
    #northside_canal_nodes = [ 12246878 ]


    assert len(northside_canal_nodes) == len(southside_canal_nodes), "%d vs %s" % (len(northside_canal_nodes), len(southside_canal_nodes))

    return zip(northside_canal_nodes, southside_canal_nodes)


def download_osm_data(output_file):
    top = 53.3635
    bottom = 53.3269
    left = -6.3161
    right = -6.2115

    url = "http://www.overpass-api.de/api/map?bbox=%s,%s,%s,%s" % (left, bottom, right, top)
    if not os.path.exists(output_file):
        urllib.urlretrieve(url, output_file)

@contextmanager
def print_status(start, end="done"):
    print start
    yield
    print end

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def circle_around(x0, y0, radius_m, num_points=20):
    """
    Returns: [ [x, y], [x, y], ... ] a set of num_points that form a circle around the point x, y
    """
    degree_lat_m, degree_lon_m = degree_size_m(lat=x0, lon=y0)
    result = []
    for rad in drange(0, 2*math.pi, (2*math.pi)/num_points):
        x, y = math.cos(rad), math.sin(rad)
        result.append((x0 + (radius_m/degree_lat_m)*x, y0 + (radius_m/degree_lon_m)*y))

    return result

def circles_around_nodes(node_ids, nodes, distance_m):
    results = []
    for node_id in node_ids:
        node = nodes[node_id]
        results.append(circle_around(node['lon'], node['lat'], distance_m))
    return results

def circles_around_nodes_wkt(node_ids, nodes, distance_m):
    circles = circles_around_nodes(node_ids, nodes, distance_m)
    return "MULTILINESTRING("+", ".join("("+", ".join(["%s %s" % point for point in circle]+["%s %s" % circle[0]]) + ")" for circle in circles)+")"

def ensure_all_nodes_exists(all_nodes, transdublin_points):
    for node1, node2 in transdublin_points:
        assert node1 in all_nodes, node1
        assert node2 in all_nodes, node2

def main():
    osm_file = "dublin-inside-canals.osm"
    with print_status("Downloading data"):
        download_osm_data(osm_file)

    with print_status("Parsing OSM file"):
        nodes, ways, nodes_ways, ways_ways, node_connections, pubs = parse_osm_file(osm_file)

    print "The pubs that are known:"
    print "GEOMETRYCOLLECTION("+",".join("POINT(%f %f)" % (nodes[pub]['lon'], nodes[pub]['lat']) for pub in pubs)+")"
    print "Pubs"
    print len(pubs)

    transdublin_points = one_side_of_dublin_to_the_other()
    ensure_all_nodes_exists(nodes, transdublin_points)

    print "Connections:"
    print "GEOMETRYCOLLECTION("+",".join("LINESTRING(%f %f, %f %f)" % (nodes[nd1]['lon'], nodes[nd1]['lat'], nodes[nd2]['lon'], nodes[nd2]['lat']) for nd1, nd2 in transdublin_points)+")"

    routes = []
    original_node_connections = deepcopy(node_connections)
    for distance in range(35, 1, -1):
        node_connections = deepcopy(original_node_connections)
        routes_at_start = len(routes)
        print "Trying with a buffer of %dm" % distance
        print "The buffer areas of %sm around the pubs" % (distance)
        print circles_around_nodes_wkt(pubs, nodes, distance)


        with print_status("Removing nodes that are within %dm of a pub..."%distance):
            node_connections = remove_node_connections_close_to_pub( nodes, pubs, node_connections, distance_sqrd=(distance*distance))

        # produce a list of the nodes at the end
        end_points = set()
        end_points = end_points.union(nd1 for nd1, nd2 in transdublin_points)
        end_points = end_points.union(nd2 for nd1, nd2 in transdublin_points)

        print "Num connections ", len(node_connections)
        node_connections = remove_dead_ends(node_connections, end_points)
        print "after removing dead ends Num connections ", len(node_connections)

        print "This is the paths we can take:"
        print all_node_connections_wkt(node_connections, nodes)

        print "About to look for a route, there are ", sum(len(x) for x in node_connections.values())/2, " node connections"
        for start, end in transdublin_points:
            print "Distance to cover: LINESTRING(%s %s, %s %s)" % (nodes[start]['lon'], nodes[start]['lat'], nodes[end]['lon'], nodes[end]['lat'])
            for route in itertools.islice(find_route(start, end, nodes, node_connections, keep_n_closest_routes=500, max_steps=10000000), 5):
                if len(routes) == 0:
                    print "Found a route! distance = %d" % distance
                routes.append(route)
                coords = [(nodes[x]['lon'], nodes[x]['lat']) for x in route]
                print "LINESTRING("+", ".join("%s %s" % x for x in coords)+")"

        print "Found %d routes for distance = %d" % (len(routes) - routes_at_start, distance)
        routes.sort(key=lambda x: len(x))
        print "Here's our routes: "
        for route in routes:
            print len(route)
            coords = [(nodes[x]['lon'], nodes[x]['lat']) for x in route]
            print "LINESTRING("+", ".join("%s %s" % x for x in coords)+")"




    print "done"

def all_node_connections_wkt(node_connections, nodes):
    seen_nodes = set()
    linestring_parts = []
    for n1 in node_connections:
        for n2 in node_connections[n1]:
            if n1 < n2:
                node1, node2 = n1, n2
            else:
                node1, node2 = n2, n1
            if (node1, node2) in seen_nodes:
                continue
            seen_nodes.add((node1, node2))
            linestring_parts.append("(%f %f, %f %f)" % (nodes[node1]['lon'], nodes[node1]['lat'], nodes[node2]['lon'], nodes[node2]['lat']))

    return "MULTILINESTRING(" + ", ".join(linestring_parts) + ")"


def route_length(route, nodes):
    return sum( flat_earth_distance(lon1=nodes[route[i]]['lon'], lat1=nodes[route[i]]['lat'], lon2=nodes[route[i+1]]['lon'], lat2=nodes[route[i+1]]['lat']) for i in range(len(route)-2) )

def sort_routes(routes, nodes):
    routes_with_length = []
    for route in routes:
        heappush(routes_with_length, ( route_length(route, nodes), route) )

    return [x[1] for x in routes_with_length]


def remove_dead_ends(nc, keep_these_nodes):
    # In order to speed things, up let's remove all dead ends. This is nodes
    # that don't connection to anything else. However to prevent ourselves from
    # removing the start and end points, we keep all the nodes in
    # keep_these_nodes

    def has_dead_end(nc, keep_these_nodes):
        """
        Returns true is this node_connections object has a dead end. False otherwise.
        """
        for nd in nc:
            if nd not in keep_these_nodes and len(nc[nd]) <= 1:
                return True
        else:
            return False

    def dead_ends(nc, keep_these_nodes):
        """
        Iterates over all the nodes in this node_connections object that are at
        one end of a dead end.
        """
        for nd in nc:
            if len(nc[nd]) <= 1 and nd not in keep_these_nodes:
                yield nd

    print "Removing dead ends..."

    # Loop over all the dead ends and remove them. We can't do this in one
    # sweep, since there might be a node like this:
    #  ...---A-----B----C
    # C is a dead end, however B is not. After we remove C, B will now be a
    # dead end, and so we will want to remove it. Ergo we have to keep looping
    # until we no more dead ends and then stop. This will iteratively 'prune'
    # the connections
    num_connections = len(nc)
    last_num_connects = None
    while has_dead_end(nc, keep_these_nodes):

        # Loop over all the dead ends we have now. Since we'll be removing
        # items from this dictionary object, we can't use this as an iterator,
        # so we have to wait for the iterator to finish
        these_dead_ends = list(dead_ends(nc, keep_these_nodes))
        for dead_end in these_dead_ends:

            # This dead end not in it mean that we've already removed the node
            # when we removed the other end.
            if dead_end in nc:
                other_end = nc[dead_end]
                assert len(other_end) in (0, 1), "Node %s is a dead end, but is connected to %r" % (dead_end, other_end)
                del nc[dead_end]
                if len(other_end) == 1:
                    other_end = other_end.pop()
                    if other_end in nc:
                        # We might have removed this end already
                        del nc[other_end]

        assert last_num_connects is None or len(nc) < last_num_connects, "We did not remove any nodes from this loop. Before this loop there were %d connections, after this loop there are %d connections. The dead ends in this iteration are %r" % (last_num_connects, len(nc), these_dead_ends)
        last_num_connects = len(nc)


    print "done"

    return nc

def convert_node_to_radians(n):
    n['lat'] = math.radians(float(n['lat']))
    n['lon'] = math.radians(float(n['lon']))

def haversine_dist(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS* c

def degree_size_m(lon, lat):
    # how long one degree of longitude is in metre. Based on it being 20,014 km
    # from north pole to south pole and there being 180 degrees of longitude
    # between them
    degree_lon_m = (20014 * 1000 / 180.0)

    # the width of a degree of latitude depends on how far north/south we are.
    circumference_m = 40076 * 1000
    degree_lat_m = ( circumference_m * math.cos(math.radians(lon)) ) / 360.0

    return (degree_lat_m, degree_lon_m)


def flat_earth_distance(lon1, lat1, lon2, lat2):
    dlat = abs(lat1 - lat2)
    dlon = abs(lon1 - lon2)

    degree_lon_m, degree_lat_m = degree_size_m((lon1+lon2)/2.0, lat1)

    dlon_m = dlon * degree_lon_m
    dlat_m = dlat * degree_lat_m

    return (dlon_m ** 2) + (dlat_m ** 2)


def remove_node_connections_close_to_pub(nodes, pubs, node_connections, distance_sqrd):
    """
    """
    seen_connections = set()


    distances = defaultdict(list)

    copy_node_connections = deepcopy(node_connections)

    for nd1, nd2 in ((nd1, nd2) for nd1 in copy_node_connections for nd2 in copy_node_connections[nd1]):

        if nd1 not in node_connections or nd2 not in node_connections[nd1]:
            # this has been completly removed
            continue

        assert nd2 in node_connections[nd1]
        assert nd1 in node_connections[nd2]

        # make nd1 the smaller of the 2, nd2 the larger. This means we only
        # need to do one check of seen_connections
        if nd2 < nd1:
            nd1, nd2 = nd2, nd1

        if (nd1, nd2) in seen_connections:
            # we've calculated this before, since each connection is in 'both
            # ways'
            continue
        else:
            seen_connections.add((nd1, nd2))

        nd1, nd2 = nodes[nd1], nodes[nd2]
        x1, y1 = float(nd1['lon']), float(nd1['lat'])
        x2, y2 = float(nd2['lon']), float(nd2['lat'])


        for pub in pubs:
            pub = nodes[pub]


            x3, y3 = float(pub['lon']), float(pub['lat'])
            dist_sqrd = distance_sqrd_between_point_and_line(x3, y3, x1, y1, x2, y2)
            if dist_sqrd <= distance_sqrd:
                # wanna remove this item
                node_connections[nd1['id']].discard(nd2['id'])
                node_connections[nd2['id']].discard(nd1['id'])
            #heappush( distances[pub['id']], ( dist_sqrd, (nd1['id'], nd2['id']) ) )

            # Only keep the closest 200
            #distances[pub['id']] = distances[pub['id']][:200]


    return node_connections

def distance_sqrd_between_point_and_line(x, y, x1, y1, x2, y2):
    # docs: http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
    len_connection_sqrd = (x2 - x1)**2 + (y2 - y1)**2
    u = ( (x - x1)*(x2 - x1) + (y - y1)*(y2 - y1) ) / len_connection_sqrd

    if u < 0:
        dist_sqrd = flat_earth_distance(y1, x1, y, x)
    elif u > 1:
        dist_sqrd = flat_earth_distance(y2, x2, y, x)
    else:
        x0, y0 = x1 + u * (x2 - x1) , y1 + u * (y2 - y1)
        dist_sqrd = flat_earth_distance(y0, x0, y, x)

    return dist_sqrd


def connect_nodes_in_way(way, node_connections):
    if not is_routable_way(way):
        #print "Not adding way ", way['id'], " tags:", ", ".join(("%s=%s" % (key, way['tags'][key])).encode("utf8") for key in way['tags'])
        return

    nds = way['nodes']
    if way['tags'].get('area', 'no') == 'yes':
        # it's an area, so we can go from one corner to any other
        for nd1, nd2 in itertools.combinations([x['id'] for x in nds], 2):
            if nd1 != nd2:
                node_connections[nd1].add(nd2)
                node_connections[nd2].add(nd1)
    else:

        # we can only go along in a line

        # say that each node is connected to each following node
        # it is here that we should remove 'connections' if there is a pub.
        for nd1, nd2 in [(nds[i]['id'], nds[i+1]['id']) for i in range(len(nds)-1)]:
            assert nd1 != nd2
            node_connections[nd1].add(nd2)
            node_connections[nd2].add(nd1)

def is_routable_way(way):
    """
    """
    if way['tags'].get('highway', None) in ['pedestrian', 'trunk', 'trunk_link',
            'footway', 'residential', 'primary', 'secondary',
            'unclassified', 'service', 'tertiary', 'steps', 'track', 'path']:
        return way['tags'].get("access", "public") in ['public', 'yes']

# returns array of [ [nodeA, ...], ..., [..., x], [x, y], [y, ..], ..., [..., nodeB] ]
def find_route(nodeA, nodeB, nodes, node_connections, keep_n_closest_routes=0, max_steps=None):
    if nodeA == nodeB:
        yield tuple()
        return
    if nodeB in node_connections[nodeA]:
        yield ((nodeA, nodeB),)
        return

    # start from nodeA and grow out
    # we should replace this with a priority queue, where the priority is the
    # distance from the nodeB to the end point, this means it will try to
    # 'grow' the routes towards nodeB and will hopefully find a route sooner
    going_out = []

    # grow towards the end
    # we might not need this, this is an optimisation
    #coming_in = set()

    nodeB_lat, nodeB_lon = float(nodes[nodeB]['lat']), float(nodes[nodeB]['lon'])
    def dist(n):
        return (nodeB_lat - float(nodes[n]['lat']))**2 + (nodeB_lon - float(nodes[n]['lon']))**2


    # inital data, grow out from the start
    for x in node_connections[nodeA]:
        heappush(going_out, ( dist(x), 1 , [nodeA, x], set([(nodeA, x)])) )

    # add in a set for outside edge of the 'cloud' that we are growing out, so that we can know if we hit the goal.
    steps = 0
    while len(going_out) > 0:
        if max_steps and steps > max_steps:
            print "Giving up finding a route after %d steps" % steps
            return
        steps += 1
        dont_care, still_dont_care, current_route, visited_combinations = heappop(going_out)
        end_point = current_route[-1]

        # nodes we can go from here:
        from_here = set(x for x in node_connections[end_point] if x not in visited_combinations and x != end_point)
        new_routes = [current_route+[x] for x in from_here]

        for new in new_routes:
            if new[-1] == nodeB:
                yield new
            else:
                heappush(going_out, (dist(new[-1]), len(new), new, visited_combinations.union(set([new[-1]]))  )  )

        if keep_n_closest_routes > 0:
            going_out = going_out[:keep_n_closest_routes]


    return

main()
