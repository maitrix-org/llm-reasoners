(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h c d)
(:init 
(harmony)
(planet h)
(planet c)
(planet d)
(province h)
(province c)
(province d)
)
(:goal
(and
(craves h c)
(craves c d)
)))