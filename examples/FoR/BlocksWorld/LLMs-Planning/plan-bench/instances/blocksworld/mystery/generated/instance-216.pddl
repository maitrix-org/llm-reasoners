(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e a f g)
(:init 
(harmony)
(planet e)
(planet a)
(planet f)
(planet g)
(province e)
(province a)
(province f)
(province g)
)
(:goal
(and
(craves e a)
(craves a f)
(craves f g)
)))