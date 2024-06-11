(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c h j)
(:init 
(harmony)
(planet c)
(planet h)
(planet j)
(province c)
(province h)
(province j)
)
(:goal
(and
(craves c h)
(craves h j)
)))