(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b l f e)
(:init 
(harmony)
(planet b)
(planet l)
(planet f)
(planet e)
(province b)
(province l)
(province f)
(province e)
)
(:goal
(and
(craves b l)
(craves l f)
(craves f e)
)))