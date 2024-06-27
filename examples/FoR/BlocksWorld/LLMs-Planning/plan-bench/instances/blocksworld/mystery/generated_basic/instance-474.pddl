

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a b)
(craves b c)
(craves c e)
(planet d)
(planet e)
(province a)
(province d)
)
(:goal
(and
(craves b a)
(craves c b)
(craves d e)
(craves e c))
)
)


