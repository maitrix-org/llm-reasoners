(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i c j g)
(:init 
(harmony)
(planet i)
(planet c)
(planet j)
(planet g)
(province i)
(province c)
(province j)
(province g)
)
(:goal
(and
(craves i c)
(craves c j)
(craves j g)
)))