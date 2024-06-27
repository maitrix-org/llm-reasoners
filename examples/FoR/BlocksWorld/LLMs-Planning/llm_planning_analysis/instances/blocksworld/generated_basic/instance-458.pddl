

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b c)
(on c a)
(ontable d)
(ontable e)
(clear b)
(clear d)
(clear e)
)
(:goal
(and
(on a c)
(on b e)
(on c b)
(on e d))
)
)


