(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c j g f)
(:init 
(harmony)
(planet c)
(planet j)
(planet g)
(planet f)
(province c)
(province j)
(province g)
(province f)
)
(:goal
(and
(craves c j)
(craves j g)
(craves g f)
)))