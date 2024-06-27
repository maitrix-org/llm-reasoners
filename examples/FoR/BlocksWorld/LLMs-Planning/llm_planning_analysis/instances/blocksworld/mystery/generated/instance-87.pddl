(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h e b l)
(:init 
(harmony)
(planet h)
(planet e)
(planet b)
(planet l)
(province h)
(province e)
(province b)
(province l)
)
(:goal
(and
(craves h e)
(craves e b)
(craves b l)
)))