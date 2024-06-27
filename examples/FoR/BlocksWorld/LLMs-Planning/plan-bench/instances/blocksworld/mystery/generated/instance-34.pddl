(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c d k g)
(:init 
(harmony)
(planet c)
(planet d)
(planet k)
(planet g)
(province c)
(province d)
(province k)
(province g)
)
(:goal
(and
(craves c d)
(craves d k)
(craves k g)
)))