

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a d)
(planet b)
(craves c b)
(planet d)
(craves e c)
(province a)
(province e)
)
(:goal
(and
(craves b d)
(craves d e))
)
)


