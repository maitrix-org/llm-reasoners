

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(planet b)
(craves c b)
(planet d)
(craves e d)
(province a)
(province c)
)
(:goal
(and
(craves b c)
(craves d b)
(craves e a))
)
)


