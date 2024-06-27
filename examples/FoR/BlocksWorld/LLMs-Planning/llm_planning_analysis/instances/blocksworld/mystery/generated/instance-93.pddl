(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e a j f)
(:init 
(harmony)
(planet e)
(planet a)
(planet j)
(planet f)
(province e)
(province a)
(province j)
(province f)
)
(:goal
(and
(craves e a)
(craves a j)
(craves j f)
)))