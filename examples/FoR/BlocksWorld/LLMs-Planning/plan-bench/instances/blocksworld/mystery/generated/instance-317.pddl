(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d k b g)
(:init 
(harmony)
(planet d)
(planet k)
(planet b)
(planet g)
(province d)
(province k)
(province b)
(province g)
)
(:goal
(and
(craves d k)
(craves k b)
(craves b g)
)))