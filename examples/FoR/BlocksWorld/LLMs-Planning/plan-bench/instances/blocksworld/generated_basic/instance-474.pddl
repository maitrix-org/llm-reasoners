

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a b)
(on b c)
(on c e)
(ontable d)
(ontable e)
(clear a)
(clear d)
)
(:goal
(and
(on b a)
(on c b)
(on d e)
(on e c))
)
)


