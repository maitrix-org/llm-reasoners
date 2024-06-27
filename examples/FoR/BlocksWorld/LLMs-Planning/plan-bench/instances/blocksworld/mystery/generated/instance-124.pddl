(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j a g)
(:init 
(harmony)
(planet j)
(planet a)
(planet g)
(province j)
(province a)
(province g)
)
(:goal
(and
(craves j a)
(craves a g)
)))