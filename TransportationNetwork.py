# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:37:57 2020

@author: wany105
"""
from graph import Graph
import ShareFunction as sf

class Transportation(Graph):
    ''' TRAFFIC NETWORK CLASS
        Traffic network is a combination of basic graph
        and the demands, the informations about links, paths
        and link-path incidence matrix will be generated
        after the initialization.
    '''

    def __init__(self, graph_dict = None, color = None, name = None, lat = None, lon = None, nodenum = None, edgenum = None, \
                 O = [], D = [], nodefile = None, edgefile = None, Type = None):
        Graph.__init__(self, graph_dict, color, name, lat, lon, nodenum, edgenum, nodefile, edgefile, Type)
        self.__origins = O
        self.__destinations = D
        self.__cast()

    # Override of add_edge function, notice that when an edge
    # is added, then the links and paths will changes alongside.
    # However, it doesn't matter when a vertex is added
    def add_edge(self, edge):
        Graph.add_edge(self, edge)
        self.__cast()

    def add_origin(self, origin):
        if origin not in self.__origins:
            self.__origins.append(origin)
            self.__cast()
        else:
            print("The origin %s already exists, thus has been ignored!" % origin)

    def add_destination(self, destination):
        if destination not in self.__destinations:
            self.__destinations.append(destination)
            self.__cast()
        else:
            print("The destination %s already exists, thus has been ignored!" % destination)

    def num_of_links(self):
        return len(self.__links)

    def num_of_paths(self):
        return len(self.__paths)

    def num_of_OD_pairs(self):
        return len(self.__OD_pairs)

    def __cast(self):
        """ Calculate or re-calculate the links, paths and
            Link-Path incidence matrix
        """
        if self.__origins != None and self.__destinations != None:
            # OD pairs = Origin-Destination Pairs
            self.__OD_pairs = self.__generate_OD_pairs()
            self.__links = self.edges()
            self.__paths, self.__paths_category = self.__generate_paths_by_demands()
            # LP Matrix = Link-Path Incidence Matrix
            self.__LP_matrix = self.__generate_LP_matrix()
    
    def __generate_OD_pairs(self):
        ''' Generate the OD pairs (Origin-Destination Pairs)
            by Cartesian production 
        '''
        OD_pairs = []
        for i in range(len(self.__origins)):
            OD_pairs.append([self.__origins[i], self.__destinations[i]])
#        for o in self.__origins:
#            for d in self.__destinations:
#                OD_pairs.append([o, d])
        return OD_pairs

    def __generate_paths_by_demands(self):
        """ According the demands, i.e. the origins and the
            destinations of the traffic flow, to construct a list
            of paths which are necessary for the traffic flow
            assignment model
        """ 
        paths_by_demands = []
        paths_category = []
        od_pair_index = 0
        for OD_pair in self.__OD_pairs:
            paths = self.find_all_paths(*OD_pair)
            paths_by_demands.extend(paths)
            paths_category.extend([od_pair_index] * len(paths))
            od_pair_index += 1
        return paths_by_demands, paths_category

    def __generate_LP_matrix(self):
        """ Generate the Link-Path incidence matrix Delta:
            if the i-th link is on j-th link, then delta_ij = 1,
            otherwise delta_ij = 0
        """
        import numpy as np
        n_links = self.num_of_links()
        n_paths = self.num_of_paths()
        lp_mat = np.zeros(shape= (n_links, n_paths), dtype= int)
        path_index = 0
        for path in self.__paths:
            for i in range(len(path) - 1):
                current_link = self.__get_link_from_path_by_order(path, i)
                link_index = self.__links.index(current_link)
                lp_mat[link_index, path_index] = 1
            path_index += 1
        return lp_mat
    
    def __get_link_from_path_by_order(self, path, order):
        """ Given a path, which is a list with length N, 
            search the link by order, which is a integer
            in the range [0, N-2]
        """
        if len(path) >= 2:
            if order >= 0 and order <= len(path) - 2:
                return [path[order], path[order+1]]
            else:
                raise ValueError("%d is not in the reasonale range!" % order)
        else:
            raise ValueError("%s contains only one vertex and cannot be input!" % path)

    def disp_links(self):
        ''' Print all the links in the network by order
        '''
        counter = 0
        for link in self.__links:
            print("%d : %s" % (counter, link))
            counter += 1

    def disp_paths(self):
        """ Print all the paths in order according to
            given origins and destinations
        """
        counter = 0
        for path in self.__paths:
            print("%d : %s " % (counter, path))
            counter += 1

    def LP_matrix(self):
        ''' Return the Link-Path matrix of
            current traffic network
        '''
        return self.__LP_matrix

    def OD_pairs(self):
        """ Return the origin-destination pairs of
            current traffic network
        """
        return self.__OD_pairs

    def paths_category(self):
        """ Return a list which implies the conjugacy
            between path (self.__paths) and origin-
            destinaiton pair (self.__OD_pairs)
        """
        return self.__paths_category

    def paths(self):
        """ Return the paths with respected to given
            origins and destinations 
        """
        return self.__paths
    
    def powersignal(self, psignal = None):
        """Return demands of signal at each itersection for the electricity
        powersignal: the power of a signal: W:J/h
        """
        self.psignal = psignal
        return self.psignal
        