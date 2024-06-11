(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j g a)
(:init 
(harmony)
(planet j)
(planet g)
(planet a)
(province j)
(province g)
(province a)
)
(:goal
(and
(craves j g)
(craves g a)
)))