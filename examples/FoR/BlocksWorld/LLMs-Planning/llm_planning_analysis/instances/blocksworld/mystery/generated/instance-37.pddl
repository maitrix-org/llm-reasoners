(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c d h i)
(:init 
(harmony)
(planet c)
(planet d)
(planet h)
(planet i)
(province c)
(province d)
(province h)
(province i)
)
(:goal
(and
(craves c d)
(craves d h)
(craves h i)
)))