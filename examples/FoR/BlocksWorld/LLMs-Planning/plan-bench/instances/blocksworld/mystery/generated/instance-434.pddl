(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e a b l)
(:init 
(harmony)
(planet e)
(planet a)
(planet b)
(planet l)
(province e)
(province a)
(province b)
(province l)
)
(:goal
(and
(craves e a)
(craves a b)
(craves b l)
)))