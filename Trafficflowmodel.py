from graph import Graph
from TransportationNetwork import Transportation
import numpy as np


class TrafficFlowModel:
    ''' TRAFFIC FLOW ASSIGN MODEL
        Inside the Frank-Wolfe algorithm is given, one can use
        the method `solve` to compute the numerical solution of
        User Equilibrium problem.
    '''
    def __init__(self, graph= None, origins= [], destinations= [], 
    demands= [], link_free_time= None, link_capacity= None, link_function = [], link_type = [], sig_function = [], \
    Cycle = [], Green = [], t_service = [], hd = [], network = None):       
        
        self.network = network
        
        # Initialization of parameters
        self.link_free_time = np.array(link_free_time)
        self.link_capacity = np.array(link_capacity)
        self.link_function = np.array(link_function)
        self.link_type = np.array(link_type)
        self.link_sigfun = np.array(sig_function)
        self.Cycle = np.array(Cycle)
        self.Green = np.array(Green)
        self.t_service = np.array(t_service)
        self.hd = np.array(hd)
        self.demand = np.array(demands)

        # Alpha and beta (used in performance function)
        self._alpha = 0.15
        self._beta = 4 

        # Convergent criterion
        self._conv_accuracy = 1e-5

        # Boolean varible: If true print the detail while iterations
        self.detail = False

        # Boolean varible: If true the model is solved properly
        self.solved = False

        # Some variables for contemporarily storing the
        # computation result
        self.final_link_flow = None
        self.iterations_times = None

    def __insert_links_in_order(self, links):
        ''' Insert the links as the expected order into the
            data structure `TrafficFlowModel.__network`
        '''
        first_vertice = [link[0] for link in links]
        for vertex in first_vertice:
            self.network.add_vertex(vertex)
        for link in links:
            self.network.add_edge(link)
            
        
    def solve_golden_section(self, accuracy, detail, precision):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance.

            (Implicitly) Return
            ------
            self.solved = True
        '''
        self._conv_accuracy = accuracy
        self.disp_detail(detail)
        self.set_disp_precision(precision)
        
#        if self.detail:
#            print(self.__dash_line())
#            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
#            print(self.__dash_line())
#            print(self.__dash_line())
#            print("Initialization")
#            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.network.num_of_links())
        link_flow = self.__all_or_nothing_assign(empty_flow)

        counter = 0
        while True:
            
#            if self.detail:
#                print(self.__dash_line())
#                print("Iteration %s" % counter)
#                print(self.__dash_line())
#                print("Current link flow:\n%s" % link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            auxiliary_link_flow = self.__all_or_nothing_assign(link_flow)

            # Step 3: Linear Search
            opt_theta = self.__golden_section(link_flow, auxiliary_link_flow)
#            opt_theta = self.__bisection(link_flow, auxiliary_link_flow)
            # Step 4: Using optimal theta to update the link flow matrix
            new_link_flow = (1 - opt_theta) * link_flow + opt_theta * auxiliary_link_flow

            # Print the detail if necessary
            if self.detail:
                print("Optimal theta: %.8f" % opt_theta)
#                print("Auxiliary link flow:\n%s" % auxiliary_link_flow)

            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            if self.__is_convergent(link_flow, new_link_flow):
                if self.detail:
                    print(self.__dash_line())
                self.solved = True
                self.final_link_flow = new_link_flow
                self.iterations_times = counter
                break
            else:
                link_flow = new_link_flow
                counter += 1
        
        self.report()
        self._formatted_solution()
        
    def solve_CFW(self, epsilon, accuracy, delta):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Conjugate-Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        if self.detail:
            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
            print(self.__dash_line())
            print(self.__dash_line())
            print("Initialization")
            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1

        empty_flow = np.zeros(self.network.num_of_links())
        self.link_flow = self.all_or_nothing_assign(empty_flow)

        
        counter = 0
        while True:
            
#            if self.detail:
#                print(self.__dash_line())
#                print("Iteration %s" % counter)
#                print(self.__dash_line())
#                print("Current link flow:\n%s" % self.link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow)
            if(counter == 0):
                self.conjugate_flow = self.auxiliary_link_flow
            
            #step3 : find the conjugate direction
            if(counter != 0):
                lamb = self.__conjugate_direction(self.link_flow, self.auxiliary_link_flow, self.conjugate_flow, delta)
                self.conjugate_flow = lamb*self.conjugate_flow + (1 - lamb)*self.auxiliary_link_flow
            
            # Step 3: Linear Search
            opt_theta = self.__bisection(self.link_flow, self.conjugate_flow, epsilon)
            
            # Step 4: Using optimal theta to update the link flow matrix
            self.new_link_flow = (1 - opt_theta) * self.link_flow + opt_theta * self.conjugate_flow

            # Print the detail if necessary
            if self.detail:
                print("Optimal theta: %.8f" % opt_theta)
#                print("Auxiliary link flow:\n%s" % self.auxiliary_link_flow)
            
            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            if self.__is_convergent3(accuracy):
                if self.detail:
                    print(self.__dash_line())
                self.solved = True
                self.final_link_flow = self.new_link_flow
                self.iterations_times = counter
                break
            else:
                self.link_flow = self.new_link_flow
                counter += 1
        self.report()
        self._formatted_solution()
            
    def __conjugate_direction(self, flow, auxiliary_flow, conjugate_flow, delta):
        '''Calculate the direction to get the conjugate flow
        '''
        temp1 = auxiliary_flow - flow
        temp2 = conjugate_flow - flow
        temp3 = temp1 - temp2

        Hessian = self.__Hessian(flow, self.link_free_time, self.link_capacity)
        
        N = np.matmul(np.matmul(temp2, Hessian), temp1)
        D = np.matmul(np.matmul(temp2, Hessian), temp3)
        
        if(D != 0):
            temp = N/D
            if(temp >= 0 and temp <= 1 - delta):
                theta = N/D
            else:
                theta = 1 - delta
        else:
            theta = 0
        
        return theta
        
    def __Hessian(self, flow, t0, capacity):
        '''Calculate the Hessian Matrix
        '''
        Hessian = np.zeros([len(flow), len(flow)])
        for i in range(len(flow)):
            temp = self._alpha*self._beta*t0[i]/capacity[i]
            Hessian[i, i] = temp*(flow[i]/capacity[i])**(self._beta - 1)
        
        return Hessian
    
    def __is_convergent3(self, accuracy):
        '''Check whether the current flow is the minimum flow using criterian 3
        '''
        link_time = self.link_flow_to_link_time(self.new_link_flow)
        
        denomintor = np.sum(link_time*self.new_link_flow)
        
        self.all_or_nothing_assign(self.new_link_flow)
        numerator = np.sum(self.demand*np.array(self.path_time_min_OD))
        
        
        relative_gap = (denomintor - numerator)/denomintor
#        print(relative_gap)
        if(relative_gap < accuracy):
            return True
        else:
            return False
        
    def _formatted_solution(self):
        ''' According to the link flow we obtained in `solve`,
            generate a tuple which contains four elements:
            `link flow`, `link travel time`, `path travel time` and
            `link vehicle capacity ratio`. This function is exposed 
            to users in case they need to do some extensions based 
            on the computation result.
        '''
        if self.solved:
            link_flow = self.final_link_flow
            link_time = self.link_flow_to_link_time(link_flow)
            path_time = self.link_time_to_path_time(link_time)
            link_vc = link_flow / self.link_capacity
            return link_flow, link_time, path_time, link_vc
        else:
            return None

    def report(self):
        ''' Generate the report of the result in console,
            this function can be invoked only after the
            model is solved.
        '''
        if self.solved:
            #Set up the flow matrix for the transporation network
            self.network.flow = np.zeros([self.network.Nnum, self.network.Nnum])

            # Print the input of the model
            print(self)
            
            # Print the report
            
            # Do the computation
            link_flow, link_time, path_time, link_vc = self._formatted_solution()
            
            self.link_flow = link_flow
            self.link_time = link_time
            self.path_time = path_time
            self.link_vc = link_vc

            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("PERFORMANCE OF LINKS")
            print(self.__dash_line())
            for i in range(self.network.num_of_links()):
#                print("%2d : link= %12s, flow= %8.2f, time= %8.3f, v/c= %.3f" % (i, self.network.edges()[i], link_flow[i], link_time[i], link_vc[i]))
                fnode, tnode = int(self.network.edges()[i][0]) - 1, int(self.network.edges()[i][1]) - 1
                self.network.flow[fnode, tnode] = link_flow[i]
            print(self.__dash_line())
            print("PERFORMANCE OF PATHS (GROUP BY ORIGIN-DESTINATION PAIR)")
            print(self.__dash_line())
            counter = 0
            for i in range(self.network.num_of_paths()):
                if counter < self.network.paths_category()[i]:
                    counter = counter + 1
                    print(self.__dash_line())
#                print("%2d : group= %2d, time= %8.3f, path= %s" % (i, self.network.paths_category()[i], path_time[i], self.network.paths()[i]))
            print(self.__dash_line())
        else:
            raise ValueError("The report could be generated only after the model is solved!")

    def __bisection(self, link_flow, auxiliary_link_flow, epsilon = 1e-4):
            '''Bisection method to calculate the optimal step
            '''
            theta_h = 1
            theta_l = 0
    
            while(theta_h - theta_l > epsilon):    
                theta = 0.5*(theta_l + theta_h)
                new_flow = theta*auxiliary_link_flow + (1 - theta)*link_flow
                new_time = self.link_flow_to_link_time(new_flow)
                total_time = np.sum(new_time*(auxiliary_link_flow - link_flow))
                if(total_time > 0):
                    theta_h = theta
                else:
                    theta_l = theta
            return theta
        
    def all_or_nothing_assign(self, link_flow):
        ''' Perform the all-or-nothing assignment of
            Frank-Wolfe algorithm in the User Equilibrium
            Traffic Assignment Model.
            This assignment aims to assign all the traffic
            flow, within given origin and destination, into
            the least time consuming path

            Input: link flow -> Output: new link flow
            The input is an array.
        '''
        # LINK FLOW -> LINK TIME
        link_time = self.link_flow_to_link_time(link_flow)
        self.link_time = link_time
        # LINK TIME -> PATH TIME
        path_time = self.link_time_to_path_time(link_time)

        # PATH TIME -> PATH FLOW
        # Find the minimal traveling time within group 
        # (splited by origin - destination pairs) and
        # assign all the flow to that path
        path_flow = np.zeros(self.network.num_of_paths())
        self.path_time_min_OD = []
        for OD_pair_index in range(self.network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.network.num_of_paths()):
                if self.network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index)
            sub_path_time = [path_time[ind] for ind in indice_grouped]
            min_in_group = min(sub_path_time)
            self.path_time_min_OD.append(min_in_group)
            ind_min = sub_path_time.index(min_in_group)
            target_path_ind = indice_grouped[ind_min]
            path_flow[target_path_ind] = self.demand[OD_pair_index]
#        if self.detail:
#            print("Link time:\n%s" % link_time)
#            print("Path flow:\n%s" % path_flow)
#            print("Path time:\n%s" % path_time)
        
        # PATH FLOW -> LINK FLOW
        new_link_flow = self.path_flow_to_link_flow(path_flow)

        return new_link_flow
        
    def link_flow_to_link_time(self, link_flow):
        ''' Based on current link flow, use link 
            time performance function to compute the link 
            traveling time.
            The input is an array.
        '''
        n_links = self.network.num_of_links()
        link_time = np.zeros(n_links)
        for i in range(n_links):
            link_time[i] = self.__link_time_performance(link_flow[i], self.link_free_time[i], self.link_capacity[i], self.link_function[i], \
                                                        self.link_type[i], self.link_sigfun[i], self.Cycle[i], self.Green[i], \
                                                        self.t_service[i], self.hd[i])
        return link_time

    def link_time_to_path_time(self, link_time):
        ''' Based on current link traveling time,
            use link-path incidence matrix to compute 
            the path traveling time.
            The input is an array.
        '''
        path_time = link_time.dot(self.network.LP_matrix())
        return path_time
    
    def path_flow_to_link_flow(self, path_flow):
        ''' Based on current path flow, use link-path incidence 
            matrix to compute the traffic flow on each link.
            The input is an array.
        '''
        link_flow = self.network.LP_matrix().dot(path_flow)
        return link_flow

    def _get_path_free_time(self):
        ''' Only used in the final evaluation, not the recursive structure
        '''
        path_free_time = self.link_free_time.dot(self.network.LP_matrix())
        return path_free_time

    def __link_time_performance(self, link_flow, t0, capacity, link_function, link_type, link_sigfun, \
                                Cycle, Green, t_service, hd):
        ''' Performance function, which indicates the relationship
            between flows (traffic volume) and travel time on 
            the same link, consisting of two parts: normal time + intersection delay
        '''
        t_norm = self.__link_time_performance_norm(link_flow, t0, capacity, link_function)
        t_inter = self.__link_time_performance_intersection(link_flow, link_function, link_type, link_sigfun, Cycle, Green, capacity, t_service, hd)
        value = t_norm + t_inter
#        value = t_norm
        return value
    
    def __link_time_performance_norm(self, link_flow, t0, capacity, link_function):
        ''' Performance function, which indicates the relationship
            between flows (traffic volume) and travel time on 
            the same link. According to the suggestion from Federal
            Highway Administration (FHWA) of America, we could use
            the following function:
                t = t0 * (1 + alpha * (flow / capacity))^beta
        '''
        value = t0 * (1 + self._alpha * ((link_flow/(link_function*capacity))**self._beta))  
        return value

    def __link_time_performance_intersection(self, link_flow, link_function, link_type, link_sigfun, Cycle, Green, \
                                             Capacity, t_service, hd):
        '''Time delay due to intersection: 
           bichoice, either signal delay or unsigal delay
        '''
        delay_sig = self.__link_time_performance_intersection_sig(link_flow, link_function, link_sigfun, \
                                                                  Cycle, Green, Capacity)
        delay_unsig = self.__link_time_performance_intersection_unsig(link_flow, t_service, hd)
        
        value = link_type*(link_sigfun*delay_sig + (1 - link_sigfun)*delay_unsig) + (1 - link_type)*delay_unsig

        return value
        
    def __link_time_performance_intersection_sig(self, link_flow, link_function, link_sigfun, Cycle, Green, Capacity):
        '''Time delay due to intersection with signals
           Details can be found in HCM TRB 2000: Highway capacity manual(HCM)
        '''
        temp0 = link_flow/(link_function*Capacity)
        temp1 = temp0/(link_function*Capacity)
        temp2 = 0.5*Cycle*(1-Green/Cycle)/(1 - min(1, temp0)*Green/Cycle)
        temp3 = 900/4*((temp0 - 1)**2 + (16*temp1)**0.5)
        
#        value = temp2 + temp3
        value = temp2
        
        return value
    
    def __link_time_performance_intersection_unsig(self, link_flow, t_service, hd):
        '''Time delay due to intersection with no signals, including:
            service time(s) and waiting time(s).
            Details can be found in HCM TRB 2000: Highway capacity manual(HCM)
        '''
        value = t_service + 900/4*(link_flow*hd/3600 - 1 + ((link_flow*hd/3600 - 1)**2 + link_flow*hd**2/(450*3600/4))**0.5) + 5
        return value

    def __link_time_performance_integrated(self, link_flow, t0, capacity, link_function, hd, Green, Cycle, link_type, link_sigfun, tservice):
        ''' The integrated (with repsect to link flow) form of
            aforementioned performance function.
        '''
        val1 = t0 * link_flow
        # Some optimization should be implemented for avoiding overflow
        val2 = (self._alpha * t0 * link_flow / (self._beta + 1)) * (link_flow / (link_function*capacity))**self._beta
        
#        #Calculate the derivative of the third term
#        #third term has two subterms: signal, unsignal
#        #for the signal one: 3 terms
#        #for the 1st term
#        temp0 = link_flow/(link_function*capacity)
#        if(temp0 >= 1):
#            delaysig_prime1 = 0.5*Cycle*(1 - Green/Cycle)/(1 - Green/Cycle)*link_flow
#        else:
#            A = 0.5*Cycle*(1 - Green/Cycle)
#            B = Green/Cycle/(link_function*capacity)
#            delaysig_prime1 = -A/B*np.log(1 - B*link_flow)
#        
#        #for the 2nd term
#        delaysig_prime2 = 900/4/2/(link_function*capacity)*link_flow**2 - 900/4*link_flow
#        #for the 3rd term
#        a = 1/(link_function*capacity)**2
#        b = 16/(link_function*capacity)**2 - 2/(link_function*capacity)
#        c = 1
#        delaysig_prime3_1 = (b/(4*a) + link_flow/2)*(a*link_flow**2+b*link_flow+c)**0.5
#        delaysig_prime3_2 = (4*a*c - b**2)/8/a**1.5*np.log((2*a*link_flow + b)/a**0.5 + 2*(a*link_flow**2+b*c+c)**0.5)
#        delaysig_prime3 = 900/4*(delaysig_prime3_1 + delaysig_prime3_2)
#        
#        delaysig_prime = delaysig_prime1+delaysig_prime2+delaysig_prime3
#        
#        #for the unsignal one: three terms
#        delayunsig_prime1 = tservice*link_flow
#        delayunsig_prime3 = 5*link_flow
#        
#        delayunsig_prime2_1 = 900/4*(hd/3600/2*link_flow**2 - link_flow)
#        
#        a = (hd/3600)**2
#        b = hd**2*4/(450*3600) - hd/1800
#        c = 1
#        temp1 = (b/4/a+link_flow/2)*(a*link_flow**2 + b*link_flow + c)**0.5
#        temp2 = (4*a*c - b**2)/8/a**1.5*np.log((2*a*link_flow+b)/a**0.5 + 2*(a*link_flow**2 + b*c + c)**0.5)
#        delayunsig_prime2_2 = 900/4*(temp1 + temp2)
#        
#        delayunsig_prime = delayunsig_prime1 + delayunsig_prime3 + delayunsig_prime2_1 + delayunsig_prime2_2
#        
#        
#        val3 = link_type*link_sigfun*delaysig_prime + link_type*(1 - link_sigfun)*delayunsig_prime + (1 - link_type)*delayunsig_prime
#    
#        value = val1 + val2 + val3
        value = val1 + val2
        return value
                
    def Cal_performance(self):
        """Calculate the performance of the whole transportation system
        The performance is quantified by the reciprocal of total travel time for all users and measures the overall travel efficiency of the TN
        """
        self.single_performance = 1/np.sum(self.link_flow*self.link_time)
        
        return self.single_performance
        

    def __object_function(self, mixed_flow):
        ''' Objective function in the linear search step 
            of the optimization model of user equilibrium 
            traffic assignment problem, the only variable
            is mixed_flow in this case.
        '''
        val = 0
        for i in range(self.network.num_of_links()):
            val += self.__link_time_performance_integrated(link_flow= mixed_flow[i], t0= self.link_free_time[i], capacity= self.link_capacity[i], link_function= self.link_function[i], \
                                                           hd = self.hd[i], Green = self.Green[i], Cycle = self.Cycle[i], link_type = self.link_type[i], link_sigfun = self.link_sigfun[i],\
                                                           tservice = self.t_service[i])
        return val

    def __golden_section(self, link_flow, auxiliary_link_flow, accuracy= 1e-8):
        ''' The golden-section search is a technique for 
            finding the extremum of a strictly unimodal 
            function by successively narrowing the range
            of values inside which the extremum is known 
            to exist. The accuracy is suggested to be set
            as 1e-8. For more details please refer to:
            https://en.wikipedia.org/wiki/Golden-section_search
        '''
        # Initial params, notice that in our case the
        # optimal theta must be in the interval [0, 1]
        LB = 0
        UB = 1
        goldenPoint = 0.618
        leftX = LB + (1 - goldenPoint) * (UB - LB)
        rightX = LB + goldenPoint * (UB - LB)
        while True:
            val_left = self.__object_function((1 - leftX) * link_flow + leftX * auxiliary_link_flow)
            val_right = self.__object_function((1 - rightX) * link_flow + rightX * auxiliary_link_flow)
            if val_left <= val_right:
                UB = rightX
            else:
                LB = leftX
            if abs(LB - UB) < accuracy:
                opt_theta = (rightX + leftX) / 2.0
                return opt_theta
            else:
                if val_left <= val_right:
                    rightX = leftX
                    leftX = LB + (1 - goldenPoint) * (UB - LB)
                else:
                    leftX = rightX
                    rightX = LB + goldenPoint*(UB - LB)

    def __is_convergent(self, flow1, flow2):
        ''' Regard those two link flows lists as the point
            in Euclidean space R^n, then judge the convergence
            under given accuracy criterion.
            Here the formula
                ERR = || x_{k+1} - x_{k} || / || x_{k} ||
            is recommended.
        '''
        err = np.linalg.norm(flow1 - flow2) / np.linalg.norm(flow1)
        if self.detail:
            print("ERR: %.8f" % err)
        if err < self._conv_accuracy:
            return True
        else:
            return False
    
    def disp_detail(self, disp_detail):
        ''' Display all the numerical details of each variable
            during the iteritions.
        '''
        self.detail = disp_detail

    def set_disp_precision(self, precision):
        ''' Set the precision of display, which influences only
            the digit of numerical component in arrays.
        '''
        np.set_printoptions(precision= precision)

    def __dash_line(self):
        ''' Return a string which consistently 
            contains '-' with fixed length
        '''
        return "-" * 80
    
    def __str__(self):
        string = ""
        string += self.__dash_line()
        string += "\n"
        string += "TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - PARAMS OF MODEL"
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += "LINK Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.network.num_of_links()):
            string += "%2d : link= %s, free time= %.2f, capacity= %s \n" % (i, self.network.edges()[i], self.link_free_time[i], self.link_capacity[i])
        string += self.__dash_line()
        string += "\n"
        string += "OD Pairs Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.network.num_of_OD_pairs()):
            string += "%2d : OD pair= %s, demand= %d \n" % (i, self.network.OD_pairs()[i], self.demand[i])
        string += self.__dash_line()
        string += "\n"
        string += "Path Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.network.num_of_paths()):
            string += "%2d : Conjugated OD pair= %s, Path= %s \n" % (i, self.network.paths_category()[i], self.network.paths()[i])
        string += self.__dash_line()
        string += "\n"
        string += "Link - Path Incidence Matrix:\n"
        string += self.__dash_line()
        string += "\n"
        string += str(self.network.LP_matrix())
        return string